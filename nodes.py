import torch
import comfy.patcher_extension

# =====================================================================
# 1. TeaCache 缓存核心算法类
# =====================================================================
class _MiniMaxH3Cache:
    def __init__(self, resuse_threshold=0.15, start_percent=0.0, end_percent=1.0, max_steps=2, device="auto", verbose=False):
        self.resuse_threshold = resuse_threshold 
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.max_steps = max_steps
        self.device = device
        self.verbose = verbose
        self.reset()
        
    def reset(self):
        """重置所有缓存张量、计数器与状态"""
        self.cached_residual = None
        self.cached_tensor_2 = None
        self.cnt_a = 0  # 真实重算计数
        self.cnt_b = 0  # 缓存跳过计数
        self.consecutive_skips = 0  # 当前连续跳过计数
        self.step_counter = 0 
        self.accumulated_rel_l1 = 0.0
        self.previous_feature_sig = None
        self.cached_signature = None
        self.last_seen_timestep = None
        self.total_steps = 20
        self.coefficients = self._build_coefficients_table()
        
    def _clear_tensors(self):
        """轻量级释放 GPU 显存张量"""
        self.cached_residual = None
        self.cached_tensor_2 = None
        
    def _build_coefficients_table(self):
        """构建 TeaCache 多项式拟合系数表"""
        return [[self.resuse_threshold, 0.0]]
    
    def _signature(self, cache_ranges):
        """根据音视频区间提取特征能量指纹 Signature (物理显存隔离)"""
        signatures = []
        max_dim = min(64, getattr(self, "hidden_dim", 64))
        
        if cache_ranges:
            for start, end in cache_ranges:
                length = end - start
                if length > 0:
                    stride = max(1, length // 100)
                    sliced = torch.abs(self.current_hidden_states[start:end:stride, :max_dim])
                    signatures.append(sliced.mean(dim=-1))
                    
        if not signatures:
            stride = max(1, self.current_hidden_states.shape[0] // 100)
            sig = torch.abs(self.current_hidden_states[::stride, :max_dim]).mean(dim=-1)
            return sig.clone().detach()
        
        full_signature = torch.cat(signatures, dim=0)
        return full_signature.clone().detach()
    
    def _step_info(self, step_info):
        if step_info is None:
            return None
        
        # 安全提取浮点数标量
        if isinstance(step_info, torch.Tensor):
            t = step_info.flatten()[0].item()
        elif isinstance(step_info, (int, float)):
            t = float(step_info)
        else:
            return None
        
        if self.coefficients is None:
            self.coefficients = self._build_coefficients_table()
            
        raw_coeffs = self.coefficients[0] if isinstance(self.coefficients[0], list) else self.coefficients
        coeffs = [float(x) for x in raw_coeffs]
        
        poly_val = sum(c * (t ** i) for i, c in enumerate(coeffs))
        thresh = poly_val / max(1, len(coeffs) - 1) if len(coeffs) > 1 else self.resuse_threshold
        
        return (t, thresh)
    
    def _store_residual(self, residual_tensor):
        """保存计算好的残差 Tensor，支持 CUDA OOM 自动降级至 CPU 内存"""
        residual = residual_tensor.detach().clone()
        try:
            if self.device == "cpu":
                residual = residual.to("cpu", non_blocking=True)
            self.cached_residual = residual
        except torch.cuda.OutOfMemoryError:
            self.cached_residual = residual.to("cpu", non_blocking=True)
            
    def _apply_residual(self, hidden_states):
        """跳过计算时，将缓存的残差叠加回 hidden_states，自动对齐设备和类型"""
        if self.cached_residual is None:
            return hidden_states
        
        same_device = (self.cached_residual.device == hidden_states.device)
        same_dtype = (self.cached_residual.dtype == hidden_states.dtype)
        
        if same_device and same_dtype:
            return hidden_states + self.cached_residual
        else:
            aligned_res = self.cached_residual.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                non_blocking=True
            )
            return hidden_states + aligned_res
        
    def __call__(self, args, kwargs):
        """DiT Block 单层 Hook 调度大脑"""
        original_block = kwargs.get("original_block")
        
        if isinstance(args, dict):
            img = args["img"]
            timestep = args.get("t_emb") if args.get("t_emb") is not None else args.get("timestep")
            cache_ranges = args.get("cache_ranges") or kwargs.get("cache_ranges") or []
            block_count = args.get("block_count")
            transformer_options = args.get("transformer_options", {})
        else:
            img = args[0]
            timestep = kwargs.get("t_emb") if kwargs.get("t_emb") is not None else kwargs.get("timestep")
            cache_ranges = kwargs.get("cache_ranges", [])
            block_count = kwargs.get("block_count")
            transformer_options = kwargs.get("transformer_options", {})
            
        if timestep is None:
            timestep = transformer_options.get("timestep")
            
        # 校验输入布局格式/签名 (分辨率或设备变了立刻重置)
        current_layout_sig = (img.shape, img.dtype, img.device, block_count)
        if self.cached_signature != current_layout_sig:
            self.reset()
            self.cached_signature = current_layout_sig
            
        step_res = self._step_info(timestep)
        if step_res is None:
            return original_block(args)
        
        t, threshold = step_res
        
        # 只有当 Timestep 改变时，才真正增加步数计数器！(防止 CFG 正负条件导致步数虚高)
        if self.last_seen_timestep != t:
            self.last_seen_timestep = t
            self.step_counter += 1
            
        # 安全获取总步数：优先读 transformer_options，没有则用 _SamplingScope 自动获取的 total_steps
        sigmas = transformer_options.get("sigmas")
        if sigmas is not None and len(sigmas) > 1:
            total_steps = len(sigmas) - 1
        else:
            total_steps = getattr(self, "total_steps", 20)
        
        # 进度 progress: 第 2 步 / 20 步 = 10%
        progress = self.step_counter / max(1, total_steps)
        in_accelerate_range = (self.start_percent <= progress <= self.end_percent)
        
        # 判断缓存跳过条件
        if self.cached_residual is not None and self.previous_feature_sig is not None:
            self.current_hidden_states = img.detach().clone()
            current_feature_sig = self._signature(cache_ranges)
            
            diff = torch.abs(current_feature_sig - self.previous_feature_sig).mean().item()
            denom = torch.abs(self.previous_feature_sig).mean().item() + 1e-6
            rel_l1 = diff / denom
            
            self.accumulated_rel_l1 += rel_l1
            
            is_below_thresh = (self.accumulated_rel_l1 < threshold)
            is_under_mcs = (self.consecutive_skips < self.max_steps)
            
            # 🎯 命中缓存：SKIP
            if is_below_thresh and is_under_mcs and in_accelerate_range:
                self.cnt_b += 1
                self.consecutive_skips += 1
                
                if self.verbose:
                    print(f"[MiniMaxH3-Cache] Step {self.step_counter} [🦘 SKIP] - (rel_l1: {self.accumulated_rel_l1:.4f} < thresh: {threshold:.4f})")
                    
                img_output = self._apply_residual(img)
                if isinstance(args, dict):
                    args["img"] = img_output
                return {"img": img_output}
            
            # 未跳过原因记录
            skip_reason = []
            if not is_below_thresh: 
                skip_reason.append(f"rel_l1({self.accumulated_rel_l1:.4f}) >= thresh({threshold:.4f})")
            if not is_under_mcs: 
                skip_reason.append(f"reached max_steps({self.max_steps})")
            if not in_accelerate_range: 
                skip_reason.append(f"progress({progress*100:.0f}%) outside [{self.start_percent*100:.0f}%, {self.end_percent*100:.0f}%]")
        else:
            # ⭐ 补全第 1 步的日志原因（无历史缓存）
            skip_reason = ["initial step (no cache)"]
                    
        # 🐢 统一打印 RUN 日志（包含第 1 步）
        if self.verbose:
            print(f"[MiniMaxH3-Cache] Step {self.step_counter} [🐢 RUN] - {', '.join(skip_reason)}")
            
        # 🔄 未命中缓存 / 第 1 步：真实重算 RUN
        self.cnt_a += 1
        self.consecutive_skips = 0
        self.cached_residual = None
        
        self.current_hidden_states = img.detach().clone()
        self.previous_feature_sig = self._signature(cache_ranges)
        
        start_img = img.clone()
        
        res = original_block(args)
        output_img = res["img"] if isinstance(res, dict) else res
        
        new_residual = output_img - start_img
        self._store_residual(new_residual)
        
        self.accumulated_rel_l1 = 0.0
        
        return res
                
        # 🔄 未命中缓存 / 不在加速百分比区间 / 达到熔断上限：真实重算 RUN
        self.cnt_a += 1
        self.consecutive_skips = 0
        self.cached_residual = None
    
        self.current_hidden_states = img.detach().clone()
        self.previous_feature_sig = self._signature(cache_ranges)
    
        start_img = img.clone()
    
        res = original_block(args)
        output_img = res["img"] if isinstance(res, dict) else res
    
        new_residual = output_img - start_img
        self._store_residual(new_residual)
    
        self.accumulated_rel_l1 = 0.0
    
        return res
    
    def finish(self):
        total_steps = self.cnt_a + self.cnt_b
        skipped_steps = self.cnt_b
        
        if total_steps > 0:
            try:
                speedup = total_steps / (total_steps - skipped_steps)
            except ZeroDivisionError:
                speedup = 1.0
                
            print(f"[MiniMaxH3-Cache] Skipped {skipped_steps}/{total_steps} steps ({speedup:.2f}x speedup).")
        self.reset()


# =====================================================================
# 2. 采样生命周期包裹类 (⭐ 修复 1：安全抓取 OUTER_SAMPLE 的 sigmas 总步数)
# =====================================================================
class _SamplingScope:
    def __init__(self, cache):
        self.cache = cache

    def __call__(self, sample_fn, *args, **kwargs):
        self.cache.reset()

        # 安全从 OUTER_SAMPLE 参数中抓取 sigmas 数组并设置 total_steps
        sigmas = kwargs.get("sigmas")
        if sigmas is None:
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.ndim == 1 and len(arg) > 1:
                    sigmas = arg
                    break

        if sigmas is not None:
            self.cache.total_steps = max(1, len(sigmas) - 1)
        else:
            raise ValueError('Unable to get sigma!')

        try:
            return sample_fn(*args, **kwargs)
        finally:
            self.cache.finish()


# =====================================================================
# 3. ComfyUI 前端自定义节点类 (完全保持原样)
# =====================================================================
class MiniMaxH3Cache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "resuse_threshold": ("FLOAT", {
                    "default": 0.10,
                    "min": 0.00,
                    "max": 1.00,
                    "step": 0.01,
                    "precision": 2
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.00,
                    "max": 1.00,
                    "step": 0.01,
                    "precision": 2
                }),
                "end_percent": ("FLOAT", {
                    "default": 0.90,
                    "min": 0.00,
                    "max": 1.00,
                    "step": 0.01,
                    "precision": 2
                }),
                "max_steps": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Cache up to N consecutive steps"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto"
                }),
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "main"
    CATEGORY = "MiniMaxH3/Cache"

    def main(self, model, resuse_threshold, start_percent, end_percent, max_steps, device, verbose):
        m = model.clone()
        diffusion_model = m.model.diffusion_model
        
        if not getattr(diffusion_model, "_is_minimax_patched", False):
            del m
            raise RuntimeError("Warning: Failed to inject model patch!")
            
        if diffusion_model.__class__.__name__ != "MiniMaxH3Model":
            del m
            raise ValueError(f"Warning: Target model is {diffusion_model.__class__.__name__}, not MiniMaxH3!")

        cache = _MiniMaxH3Cache(
            resuse_threshold=resuse_threshold,
            start_percent=start_percent,
            end_percent=end_percent,
            max_steps=max_steps,
            device=device,
            verbose=verbose
        )
        scope = _SamplingScope(cache)

        # 挂载 block_loop 补丁 (接管 model.py 中的 Transformer 50 层循环)
        m.set_model_patch_replace(cache, "dit", "block_loop", 0)

        # 挂载 wrapper 采样包裹器 (自动处理 reset 和 finish)
        m.add_wrapper(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, scope)

        return (m,)
