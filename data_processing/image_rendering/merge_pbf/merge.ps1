$pbfDir = "D:\projects\maps\data\pbf\city"
$outPbf = "D:\projects\maps\data\pbf\merged\merged.pbf"

New-Item -ItemType Directory -Force -Path (Split-Path $outPbf)

function To-WSL($path) {
    $linuxPath = $path -replace "\\", "/"
    $linuxPath = $linuxPath -replace "^D:", "/mnt/d"
    return $linuxPath
}

$outWsl = To-WSL $outPbf

$pbfFiles = Get-ChildItem $pbfDir -Filter *.pbf | ForEach-Object {
    To-WSL $_.FullName
}

$inputs = $pbfFiles -join " "

$cmd = "osmium merge $inputs -o $outWsl"

Write-Host "RUN:"
Write-Host $cmd

wsl bash -c "$cmd"