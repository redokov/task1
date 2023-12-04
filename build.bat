chcp 866
powershell.exe -command .\build.ps1 ^
-md "%1" ^
-template ".\docx\template.docx" ^
-docx .\docx\%2` Версия` %3.docx ^
