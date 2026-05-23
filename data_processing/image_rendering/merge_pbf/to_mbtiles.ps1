$java = "C:\Program Files\Java\jdk-21.0.10\bin\java.exe"
$jar = "D:\projects\maps\data_processing\image_rendering\merge_pbf\planetiler.jar"

$input = "D:\projects\maps\data\pbf\merged\merged.pbf"
$out = "D:\projects\maps\data\mbtiles\out.mbtiles"

& $java -jar $jar `
    --osm-path=$input `
    --output=$out `
    --download