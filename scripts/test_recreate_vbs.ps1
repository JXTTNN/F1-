﻿# 检查VBS文件编码并对比成功和失败的VBS
$distDir = "D:\F1OPT-Test\dist"

# 检查正式VBS编码
$bytes = [System.IO.File]::ReadAllBytes("$distDir\F1OPT桌面启动.vbs")
Write-Host "正式VBS前20字节:" -ForegroundColor Yellow
$bytes[0..19] | ForEach-Object { Write-Host ("{0:X2}" -f $_) -NoNewline; Write-Host " " -NoNewline }
Write-Host ""

# 重新创建VBS脚本（确保ASCII编码）
$vbsContent = @"
' F1OPT 桌面快捷启动脚本
Set WshShell = CreateObject("WScript.Shell")
Set Fso = CreateObject("Scripting.FileSystemObject")
scriptDir = Fso.GetParentFolderName(WScript.ScriptFullName)
exePath = scriptDir & "\F1OPT.exe"
url = "http://127.0.0.1:8000"

If Not Fso.FileExists(exePath) Then
    MsgBox "未找到 F1OPT.exe", vbExclamation, "F1OPT"
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

cmdLine = "cmd /c start /B " & Chr(34) & Chr(34) & " " & Chr(34) & exePath & Chr(34)
WshShell.Exec cmdLine

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
    MsgBox "F1OPT启动超时", vbExclamation, "F1OPT"
End If
"@

# 先清理
$f1optOld = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optOld) {
    $f1optOld | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
    Write-Host "已清理旧进程" -ForegroundColor Yellow
}

# 写入新VBS（ASCII编码）
$newVbsPath = "$distDir\F1OPT桌面启动.vbs"
$vbsContent | Out-File -FilePath $newVbsPath -Encoding ASCII
Write-Host "VBS已重新写入（ASCII编码）" -ForegroundColor Green

# 检查新VBS编码
$bytes2 = [System.IO.File]::ReadAllBytes($newVbsPath)
Write-Host "新VBS前20字节:" -ForegroundColor Yellow
$bytes2[0..19] | ForEach-Object { Write-Host ("{0:X2}" -f $_) -NoNewline; Write-Host " " -NoNewline }
Write-Host ""

# 测试新VBS
Write-Host ""
Write-Host "[1] 运行新VBS脚本..." -ForegroundColor Yellow
$startTime = Get-Date
Start-Process -FilePath "wscript.exe" -ArgumentList "`"$newVbsPath`"" -NoNewWindow

# 等待API
$apiReady = $false
for ($i = 1; $i -le 30; $i++) {
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

# 检查进程
Write-Host ""
Write-Host "[2] 检查 F1OPT 进程..." -ForegroundColor Yellow
$f1opt = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt) {
    Write-Host "    ✅ F1OPT PID: $($f1opt.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT未运行" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== 测试完成 ===" -ForegroundColor Cyan