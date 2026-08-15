$ErrorActionPreference = 'Stop'
$inp = @"
protocol=https
host=github.com

"@
$tok = ($inp | git credential fill 2>$null | ForEach-Object { $_.Trim() } | Where-Object { $_ -like 'password=*' } | Select-Object -First 1).Substring(9)
if (-not $tok) { Write-Output 'NO_TOKEN'; exit 1 }
Write-Output ('token_prefix=' + $tok.Substring(0,6) + ' len=' + $tok.Length)
$url = "https://uploads.github.com/repos/JXTTNN/F1-/releases/371120898/assets?name=f1opt-v1.2.0-win64.zip"
curl.exe -sS --ssl-no-revoke --retry 3 --retry-delay 5 --retry-all-errors --connect-timeout 30 --max-time 900 -X POST -H "Authorization: Bearer $tok" -H "Content-Type: application/zip" -H "User-Agent: f1opt" --data-binary "@dist\f1opt-v1.2.0-win64.zip" $url
Write-Output ('UPLOAD_EXIT=' + $LASTEXITCODE)
