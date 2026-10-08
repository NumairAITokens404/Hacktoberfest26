# Vitalis MRI API

Compact Vercel-ready FastAPI service for research-only 2D brain MRI classification.
It uses an ONNX export of the `ThisenEkanayake/brain-tumor-detection`
ResNet-18 checkpoint and does not bundle PyTorch.

## Deploy

1. Create a new Vercel project from this repository.
2. Set the project's root directory to `mri-api`.
3. Deploy and confirm `GET /health` returns `{"status":"ok",...}`.
4. Set the main Vitalis deployment's `MRI_API_URL` to
   `https://<your-mri-project>.vercel.app/mri`.

`POST /mri` accepts JSON containing `mime_type`, `image_base64`, and an optional
`health_report`. This is a research prototype, not a medical device.
