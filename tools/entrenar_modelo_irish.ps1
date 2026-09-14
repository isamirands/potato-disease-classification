[CmdletBinding()]
param(
    [string]$ReferenceRun = "artifacts/runs/20260824T043217Z-sin-sesiones",
    [string]$CampisRun = "artifacts/runs/20260907T001939Z-modelo1-solo-campis",
    [string]$MixtoLowRun = "artifacts/runs/20260907T010532Z-modelo2-mitad-campis-irish",
    [string]$Remote = "origin"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$python = Join-Path $projectRoot ".venv/Scripts/python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "No se encontro el entorno virtual: $python"
}
foreach ($run in @($ReferenceRun, $CampisRun, $MixtoLowRun)) {
    if (-not (Test-Path -LiteralPath $run)) {
        throw "No se encontro la corrida requerida para la comparacion: $run"
    }
}

& $python tools/preparar_experimentos.py --reference-run $ReferenceRun
if ($LASTEXITCODE -ne 0) { throw "No se pudo crear el manifest de Modelo_IRISH." }

& $python run.py train --config configs/modelo3_solo_irish.toml --run-name modelo3-solo-irish --verbose
$trainExit = $LASTEXITCODE

$irishRun = Get-ChildItem artifacts/runs -Directory -Filter "*-modelo3-solo-irish" |
    Where-Object { $_.Name -notlike "*-50" } |
    Sort-Object Name -Descending |
    Select-Object -First 1
if ($null -eq $irishRun -or -not (Test-Path -LiteralPath (Join-Path $irishRun.FullName "model.tflite"))) {
    throw "El entrenamiento no produjo model.tflite. Codigo de salida: $trainExit"
}
if ($trainExit -ne 0) {
    Write-Warning "El pipeline devolvio codigo $trainExit, pero el .tflite existe. Se continuara con la evaluacion completa; revisa tflite_parity.json."
}

& $python tools/evaluar_tflite.py --run $irishRun.FullName --threads 1 --batch-size 32
if ($LASTEXITCODE -ne 0) { throw "Fallo la evaluacion TFLite completa." }

& $python tools/generar_datos_comparacion.py `
    --reference $ReferenceRun `
    --model1 $CampisRun `
    --model2 $MixtoLowRun `
    --model3 $irishRun.FullName `
    --output docs/comparacion-tflite/data.json
if ($LASTEXITCODE -ne 0) { throw "Fallo la actualizacion del tablero." }

$resultsToCommit = @(
    (Join-Path $irishRun.FullName "config.resolved.json"),
    (Join-Path $irishRun.FullName "evaluation"),
    (Join-Path $irishRun.FullName "evaluation_tflite"),
    (Join-Path $irishRun.FullName "figures"),
    (Join-Path $irishRun.FullName "history.csv"),
    (Join-Path $irishRun.FullName "history_head.csv"),
    (Join-Path $irishRun.FullName "history_fine_tune.csv"),
    (Join-Path $irishRun.FullName "labels.json"),
    (Join-Path $irishRun.FullName "manifest_summary.json"),
    (Join-Path $irishRun.FullName "model.labels.json"),
    (Join-Path $irishRun.FullName "model.metadata.json"),
    (Join-Path $irishRun.FullName "tflite_parity.json"),
    (Join-Path $irishRun.FullName "training_summary.json"),
    "docs/comparacion-tflite/data.json"
) | Where-Object { Test-Path -LiteralPath $_ }

git add -- $resultsToCommit
git add -f -- artifacts/manifests/protocolo_comparacion.json
git diff --cached --quiet
$gitDiffExit = $LASTEXITCODE
if ($gitDiffExit -gt 1) { throw "No se pudo comprobar el estado de Git." }
if ($gitDiffExit -eq 1) {
    git commit -m "results: add Modelo_IRISH TFLite evaluation"
    if ($LASTEXITCODE -ne 0) { throw "Fallo el commit de resultados." }
    git push $Remote HEAD
    if ($LASTEXITCODE -ne 0) { throw "Fallo la subida de resultados a $Remote." }
} else {
    Write-Host "No hay resultados nuevos para subir a Git."
}

Write-Host "`nRutas de modelos:"
Write-Host "Modelo_COMPLETO  Keras: $(Join-Path $ReferenceRun 'model.keras')"
Write-Host "Modelo_COMPLETO TFLite: $(Join-Path $ReferenceRun 'model.tflite')"
Write-Host "Modelo_CAMPIS    Keras: $(Join-Path $CampisRun 'model.keras')"
Write-Host "Modelo_CAMPIS   TFLite: $(Join-Path $CampisRun 'model.tflite')"
Write-Host "Modelo_IRISH    Keras: $(Join-Path $irishRun.FullName 'model.keras')"
Write-Host "Modelo_IRISH   TFLite: $(Join-Path $irishRun.FullName 'model.tflite')"
Write-Host "Modelo_MIXTO_LOW Keras: $(Join-Path $MixtoLowRun 'model.keras')"
Write-Host "Modelo_MIXTO_LOW TFLite: $(Join-Path $MixtoLowRun 'model.tflite')"
Write-Host "`nTablero: http://127.0.0.1:8766"
