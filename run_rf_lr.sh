cell_line_list="KM12 MCF7 MDA-MB-231_ATCC HS_578T BT-549 T-47D SF-268 SF-295 SF-539 SNB-19 SNB-75 U251 COLO_205 HCC-2998 HCT-116 HCT-15 HT29 SW-620 CCRF-CEM HL-60(TB) K-562 MOLT-4 RPMI-8226 SR LOX_IMVI MALME-3M M14 SK-MEL-2 SK-MEL-28 SK-MEL-5 UACC-257 UACC-62 MDA-MB-435 MDA-N A549_ATCC EKVX HOP-62 HOP-92 NCI-H226 NCI-H23 NCI-H322M NCI-H460 NCI-H522 IGROV1 OVCAR-3 OVCAR-4 OVCAR-5 OVCAR-8 SK-OV-3 NCI_ADR-RES PC-3 DU-145 786-0 A498 ACHN CAKI-1 RXF_393 SN12C TK-10 UO-31"
for cell_line in $cell_line_list; do
    for test_fold in {1..7}; do
        for seed in {1..5}; do    
            python sklearn_models.py --cell_line $cell_line --test_fold $test_fold --seed $seed --split scaffold --search no_tune
        done
    done
done