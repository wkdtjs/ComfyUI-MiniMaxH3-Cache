import torch
import comfy.model_prefetch
import comfy.ldm.minimax.model as minimax_model

from .nodes import MiniMaxH3Cache

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3Cache": MiniMaxH3Cache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3Cache": "MiniMaxH3 Cache",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def apply_minimax_h3_patch():
    """给 comfy.ldm.minimax.model.MiniMaxH3Model 动态注入修改版的功能"""
    
    MiniMaxH3Model = getattr(minimax_model, "MiniMaxH3Model", None)
    if MiniMaxH3Model is None:
        print('[MiniMaxH3-Cache] Warning: "MiniMaxH3Model" not found in "comfy.ldm.minimax.model"')
        return
    
    # 1. 新增辅助方法: _run_blocks
    def _run_blocks(self, h, t_emb, mod_segments, rope_freqs, transformer_options, start=0, end=None):
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        end = len(self.blocks) if end is None else end
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks[start:end]), h.device, transformer_options)
        for i in range(start, end):
            block = self.blocks[i]
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, h.device, block)
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                         transformer_options=args["transformer_options"])}
                h = blocks_replace[("double_block", i)](
                    {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                     "transformer_options": transformer_options},
                    {"original_block": block_wrap})["img"]
            else:
                h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
        if prefetch_queue is not None:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, h.device, None)
        return h
    
    # 2. 替换方法: _forward
    def patched_forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        video_x, audio_x = x[0], x[1]
        orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        video_x = minimax_model.comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
        if video_x.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        payload = minimax_payload or {}
        device = video_x.device
        dtype = context.dtype  # compute dtype
        
        latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        audio_t = audio_x.shape[-1]
        text_len = context.shape[1]
        
        layout = payload.get("layout")
        if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
            layout = minimax_model.PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                                keyframes=payload.get("keyframes"),
                                                refs=payload.get("refs"),
                                                frame_count=payload.get("frame_count"))
            
        shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
        shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - minimax_model.time_shift_sigma(sigma_v, shift_v, shift_a))
        
        vis_aug = float(payload.get("visual_cond_noise_aug", minimax_model.VISUAL_COND_TIMESTEP))
        aud_aug = float(payload.get("audio_cond_noise_aug", minimax_model.AUDIO_COND_TIMESTEP))
        has_vis_cond = any(k in ("cond", "ref_img") for _, _, k in layout.segments)
        has_aud_cond = any(k == "ref_audio" for _, _, k in layout.segments)
        seg_t = {"text": t_v, "video": t_v, "audio": t_a,
                 "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
                 "ref_audio": max(t_a, aud_aug)}
        unique_t = sorted({t_v, t_a} | ({seg_t["cond"]} if has_vis_cond else set())
                          | ({seg_t["ref_audio"]} if has_aud_cond else set()))
        t_row = {t: i for i, t in enumerate(unique_t)}
        seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}
        
        text_tags = payload.get("text_token_tags")
        mod_segments = []
        for a, b, kind in layout.segments:
            row_base = t_row[seg_t[kind]] * 3
            if kind == "text" and text_tags is not None:
                tags = text_tags.view(-1).tolist()
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                        run_start = i
            else:
                mod_segments.append((a, b, row_base + seg_tag[kind]))
                
        img_update = layout.img_update.to(device)
        audio_update = layout.audio_update.to(device)
        video_rows = minimax_model.patchify_video(video_x.to(torch.float32), self.patch_size)
        audio_rows = minimax_model.pack_audio(audio_x.to(torch.float32))
        cond_video_rows = self._cond_video_rows(payload, device)
        cond_audio_rows = self._cond_audio_rows(payload, device)
        
        all_video_rows = video_rows
        if cond_video_rows is not None:
            all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
            all_video_rows[~img_update] = cond_video_rows
            all_video_rows[img_update] = video_rows
        all_audio_rows = audio_rows
        if cond_audio_rows is not None:
            all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
            all_audio_rows[~audio_update] = cond_audio_rows
            all_audio_rows[audio_update] = audio_rows
            
        video_embed = self.video_patch_proj(all_video_rows).to(dtype)
        audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
        text_states = context[0]
        if text_states.shape[-1] != self.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states),
                                             transformer_options=transformer_options)
            
        h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
        voff = aoff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                h[a:b] = text_states
            elif kind in ("cond", "ref_img", "video"):
                h[a:b] = video_embed[voff:voff + n]
                voff += n
            else:
                h[a:b] = audio_embed[aoff:aoff + n]
                aoff += n
                
        t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
        if self.use_adaln_curves:
            table = minimax_model.comfy.model_management.cast_to(self.adaln_t_table, device=device)
            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
            t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
        else:
            t_emb = self.time_embedder(t_vals).to(dtype)
            
        rope_freqs = minimax_model.rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)
        
        # ---------------- 修改版核心改动 ----------------
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        cache_ranges = [(a, b) for a, b, kind in layout.segments if kind in ("audio", "video")]
        if ("block_loop", 0) in blocks_replace:
            def block_loop_wrap(args):
                return {"img": self._run_blocks(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                                args["transformer_options"], args.get("start", 0), args.get("end"))}
            h = blocks_replace[("block_loop", 0)](
                {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                 "transformer_options": transformer_options, "cache_ranges": cache_ranges, "block_count": len(self.blocks)},
                {"original_block": block_loop_wrap})["img"]
        else:
            h = self._run_blocks(h, t_emb, mod_segments, rope_freqs, transformer_options)
        # ------------------------------------------------
            
        video_seg = next((a, b, t_row[seg_t["video"]]) for a, b, k in layout.segments if k == "video")
        audio_seg = next((a, b, t_row[seg_t["audio"]]) for a, b, k in layout.segments if k == "audio")
        v, a = self.final_layer(h, t_emb, video_seg, audio_seg)
        
        video_out = minimax_model.unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        audio_out = minimax_model.unpack_audio(a)
        
        slope_a = minimax_model.time_shift_slope(sigma_v, shift_v, shift_a).to(audio_out.dtype)
        return [-video_out.to(video_x.dtype), (-slope_a) * audio_out.to(audio_x.dtype)]
    
    # 替换绑存方法
    MiniMaxH3Model._run_blocks = _run_blocks
    MiniMaxH3Model._forward = patched_forward
    MiniMaxH3Model._is_minimax_patched = True
    print('[MiniMaxH3-Cache] Applied patch to "comfy.ldm.minimax.model.MiniMaxH3Model"')
    
# 在模块加载时自动运行打补丁
apply_minimax_h3_patch()