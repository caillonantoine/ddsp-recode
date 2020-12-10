runs="runs/johngarner_slow runs/sax_full runs/sol_ordinario"

mkdir temp

for run in $runs;
do
    echo exporting run $(basename $run)
    python export.py --out-dir temp/$(basename $run) --data true --run $run
done

echo creating archive
cd temp
tar -czf ../ddsp.tar.gz *
cd ../
rm -fr temp