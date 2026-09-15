﻿# 测试 VBS 中 WshShell.Exec 方案（用CreateProcess而非ShellExecute）
$distDir = "D:\F1OPT-Test\dist"

Write-Host "=== VBS WshShell.Exec 测试 ===" -ForegroundColor Cyan

# 先清理
$f1optOld = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optOld) {
    $f1optOld | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
    Write-Host "已清理旧进程" -ForegroundColor Yellow
}

# VBS脚本：用 WshShell.Exec 而不是 WshShell.Run
# Exec 用 CreateProcess，继承标准流
$testVbs = @"
Set WshShell = CreateObject("WScript.Shell")
Set Fso = CreateObject("Scripting.FileSystemObject")
scriptDir = Fso.GetParentFolderName(WScript.ScriptFullName)
exePath = scriptDir & "\F1OPT.exe"
url = "http://127.0.0.1:8000"

' 检查API是否已在运行
Set http = CreateObject("MSXML2.XMLHTTP")
On Error Resume Next
http.Open "GET", url & "/api/v1/health", False
http.Send
If Err.Number = 0 And http.Status = 200 Then
    WshShell.Run url
    WScript.Quit 0
End If
On Error GoTo 0

' 设置工作目录
WshShell.CurrentDirectory = scriptDir

' 用 WshShell.Exec 启动（CreateProcess方式，继承标准流）
Set exec = WshShell.Exec("cmd /c start /B " & Chr(34) & Chr(34) & " " & Chr(34) & exePath & Chr(34))

' 等待API就绪
Set http2 = CreateObject("MSXML2.XMLHTTP")
ready = False
For i = 1 To 30
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
$testVbsPath = "$distDir\test_exec_vbs.vbs"
$testVbs | Out-File -FilePath $testVbsPath -Encoding ASCII

Write-Host "[1] VBS WshShell.Exec 方案..." -ForegroundColor Yellow
$startTime = Get-Date
Start-Process -FilePath "wscript.exe" -ArgumentList "`"$testVbsPath`"" -NoNewWindow

# 等待API
$apiReady = $false
for ($i = 1; $i -le 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.code -eq 200) {
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

# 清理测试VBS
Remove-Item $testVbsPath -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== 测试完成 ===" -ForegroundColor Cyan