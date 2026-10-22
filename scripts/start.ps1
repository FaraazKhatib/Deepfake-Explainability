param([ValidateSet('balanced', 'oct08', 'previous', 'legacy', 'candidate')][string]$Profile = 'balanced')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
# These variables are scoped to this PowerShell process. Stop the server and
# launch legacy to revert model selection without touching source files.
if ($Profile -eq 'legacy') {
    $env:VT_MODELS = 'deepfake_wav2vec2,xlsr_deepfake,wavlm_itw,ast_asvspoof,deepfake_voice_detector,wav2vec2_base_itw,deepfake_audio_hemgg,distilhubert_itw,bisher_asv,zeyadd_deepfake_v1'
    $env:MODEL_NAME = 'dima806/deepfake_vs_real_image_detection'
    $env:FC_NLI_MODEL = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
} else {
    $env:VT_MODELS = 'detectra_openfake,wavlm_itw'
    if ($Profile -eq 'oct08') { $env:VT_MODELS = 'swin_image,wavlm_itw' }
    $env:MODEL_NAME = 'dima806/deepfake_vs_real_image_detection'
    $env:FC_NLI_MODEL = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
    if ($Profile -eq 'previous') {
        $env:VT_MODELS = 'wavlm_itw,bisher_asv'
    }
    if ($Profile -eq 'candidate') {
        $env:VT_MODELS = 'siglip_image,siglip_face,wavlm_itw,bisher_asv'
        $env:MODEL_NAME = 'prithivMLmods/Deepfake-Detect-Siglip2'
        $env:FC_NLI_MODEL = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
    }
}
Push-Location (Join-Path $projectRoot 'backend')
try { & (Join-Path $projectRoot '.venv/Scripts/python.exe') -m uvicorn app.main:app --host 127.0.0.1 --port 8001 }
finally { Pop-Location }
