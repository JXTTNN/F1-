﻿# 用正确编码和直接拼接方式写入VBS
$distDir = "D:\F1OPT-Test\dist"

# 先清理
$f1optOld = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optOld) {
    $f1optOld | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
    Write-Host "已清理旧进程" -ForegroundColor Yellow
}

# VBS脚本内容（直接在Exec参数中拼接，不用变量）
$vbsContent = @"
' F1OPT Desktop Launcher
Set WshShell = CreateObject("WScript.Shell")
Set Fso = CreateObject("Scripting.FileSystemObject")
scriptDir = Fso.GetParentFolderName(WScript.ScriptFullName)
exePath = scriptDir & "\F1OPT.exe"
url = "http://127.0.0.1:8000"

If Not Fso.FileExists(exePath) Then
    MsgBox "F1OPT.exe not found", vbExclamation, "F1OPT"
    WScript.Quit 1
End If

Set http = CreateObject("MSXML2.XMLHTTP")
On Error Resume Next
http.Open "GET", url & "/api/v1/health", False
http.Send
If Err.Number = 0 And http.Status = 200 Then
    WshShell.Run url
    WScript.Quit 0
End If
On Error GoTo 0

WshShell.CurrentDirectory = scriptDir

Set exec = WshShell.Exec("cmd /c start /B " & Chr(34) & Chr(34) & " " & Chr(34) & exePath & Chr(34))

Set http2 = CreateObject("MSXML2.XMLHTTP")
ready = False
For i = 1 To 60
    WScript.Sleep 1000
    On Error Resume Next
    http2.Open "GET", url & "/api/v1/health", False
    http2.Send
    If Err.Number = 0 And http2.Status = 200 Then
        ready = True
        Exit For
    End If
    On Error GoTo 0
Next

If ready Then
    WshShell.Run url
Else
    MsgBox "F1OPT startup timeout", vbExclamation, "F1OPT"
End If
"@

# 用ASCII编码写入（和成功的简化VBS一样的编码）
$vbsPath = "$distDir\F1OPT桌面启动.vbs"
$vbsContent | Out-File -FilePath $vbsPath -Encoding ASCII
Write-Host "VBS已写入（ASCII编码，直接拼接）" -ForegroundColor Green

# 测试
Write-Host "[1] 运行VBS..." -ForegroundColor Yellow
$startTime = Get-Date
Start-Process -FilePath "wscript.exe" -ArgumentList "`"$vbsPath`"" -NoNewWindow

$apiReady = $false
for ($i = 1; $i -le 40; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) {
            $elapsed = ((Get-Date) - $startTime).TotalSeconds
            Write-Host "    ✅ API ${elapsed}秒就绪" -ForegroundColor Green
            $apiReady = $true
            break
        }
    } catch {
        if ($i % 5 -eq 0) { Write-Host "    ${i}秒..." -ForegroundColor DarkGray }
    }
}

$f1opt = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt) {
    Write-Host "    ✅ F1OPT PID: $($f1opt.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT未运行" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== 测试完成 ===" -ForegroundColor Cyan