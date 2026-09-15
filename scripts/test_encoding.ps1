﻿# 用Unicode编码写入VBS脚本并测试
$distDir = "D:\F1OPT-Test\dist"

# 先清理
$f1optOld = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optOld) {
    $f1optOld | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
    Write-Host "已清理旧进程" -ForegroundColor Yellow
}

# VBS脚本内容（含中文注释）
$vbsContent = @"
' F1OPT 桌面快捷启动脚本
' 功能：后台启动 F1OPT.exe，等待 API 就绪后打开浏览器
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

# 方案1：用Unicode编码写入
$vbsPath1 = "$distDir\F1OPT桌面启动.vbs"
$vbsContent | Out-File -FilePath $vbsPath1 -Encoding Unicode
Write-Host "方案1: Unicode编码写入" -ForegroundColor Yellow

# 检查编码
$bytes1 = [System.IO.File]::ReadAllBytes($vbsPath1)
Write-Host "前4字节: $(($bytes1[0..3] | ForEach-Object { '{0:X2}' -f $_ }) -join ' ')" -ForegroundColor Green

# 测试
Write-Host "[1] 运行Unicode编码VBS..." -ForegroundColor Yellow
$startTime = Get-Date
Start-Process -FilePath "wscript.exe" -ArgumentList "`"$vbsPath1`"" -NoNewWindow

$apiReady = $false
for ($i = 1; $i -le 25; $i++) {
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

$f1opt1 = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt1) {
    Write-Host "    ✅ F1OPT PID: $($f1opt1.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT未运行" -ForegroundColor Red
}

# 清理
if ($f1opt1) {
    $f1opt1 | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
}

# 方案2：用UTF-8 BOM编码写入
$vbsPath2 = "$distDir\test_utf8bom.vbs"
$utf8Bom = [System.Text.Encoding]::UTF8.GetPreamble()
$utf8Content = [System.Text.Encoding]::UTF8.GetBytes($vbsContent)
[System.IO.File]::WriteAllBytes($vbsPath2, $utf8Bom + $utf8Content)
Write-Host ""
Write-Host "方案2: UTF-8 BOM编码写入" -ForegroundColor Yellow

$bytes2 = [System.IO.File]::ReadAllBytes($vbsPath2)
Write-Host "前4字节: $(($bytes2[0..3] | ForEach-Object { '{0:X2}' -f $_ }) -join ' ')" -ForegroundColor Green

Write-Host "[2] 运行UTF-8 BOM编码VBS..." -ForegroundColor Yellow
$startTime2 = Get-Date
Start-Process -FilePath "wscript.exe" -ArgumentList "`"$vbsPath2`"" -NoNewWindow

$apiReady2 = $false
for ($i = 1; $i -le 25; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) {
            $elapsed = ((Get-Date) - $startTime2).TotalSeconds
            Write-Host "    ✅ API ${elapsed}秒就绪" -ForegroundColor Green
            $apiReady2 = $true
            break
        }
    } catch {
        if ($i % 5 -eq 0) { Write-Host "    ${i}秒..." -ForegroundColor DarkGray }
    }
}

$f1opt2 = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt2) {
    Write-Host "    ✅ F1OPT PID: $($f1opt2.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT未运行" -ForegroundColor Red
}

# 清理
if ($f1opt2) {
    $f1opt2 | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
}

# 方案3：纯ASCII（无中文注释）
$vbsContent3 = @"
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
    MsgBox "F1OPT startup timeout", vbExclamation, "F1OPT"
End If
"@

$vbsPath3 = "$distDir\test_ascii.vbs"
$vbsContent3 | Out-File -FilePath $vbsPath3 -Encoding ASCII
Write-Host ""
Write-Host "方案3: 纯ASCII编码（无中文）" -ForegroundColor Yellow

Write-Host "[3] 运行纯ASCII编码VBS..." -ForegroundColor Yellow
$startTime3 = Get-Date
Start-Process -FilePath "wscript.exe" -ArgumentList "`"$vbsPath3`"" -NoNewWindow

$apiReady3 = $false
for ($i = 1; $i -le 25; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) {
            $elapsed = ((Get-Date) - $startTime3).TotalSeconds
            Write-Host "    ✅ API ${elapsed}秒就绪" -ForegroundColor Green
            $apiReady3 = $true
            break
        }
    } catch {
        if ($i % 5 -eq 0) { Write-Host "    ${i}秒..." -ForegroundColor DarkGray }
    }
}

$f1opt3 = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt3) {
    Write-Host "    ✅ F1OPT PID: $($f1opt3.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT未运行" -ForegroundColor Red
}

# 清理测试文件
Remove-Item $vbsPath2, $vbsPath3 -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== 测试完成 ===" -ForegroundColor Cyan