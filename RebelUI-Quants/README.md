---
base_model: D:\AI_Tools\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\flux-2-klein-4b.safetensors
---

# RebelUI quantization

- Architecture recipe: `flux`
- Exclude: `(img_in|txt_in|time_in|vector_in|guidance_in|final_layer|pe_embedder|rope|freqs|img_mod|txt_mod|modulation|adaln|ada_ln|scale_shift|norm|embed)`
- Group size: `16`
- INT8 keys: ``
- High precision: `img_in,txt_in,time_in,vector_in,guidance_in,final_layer,pe_embedder,rope,freqs,img_mod,txt_mod,modulation,adaln,ada_ln,scale_shift,norm,embed`

Generated locally with RebelUI Auto-Quantizer. Review converter verification output for copy-param count, format histogram and measured relL2 before publishing.
