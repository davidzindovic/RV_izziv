## 1. Potrebne knjižnice:
* numpy
* opencv
* matplotlib
* math
* json

## 2. Potrebne datoteke:
* validate_two_videos.py
* main_optimized.py
* config_izziv_main.py
(za zagon kode in branje nastavitev za debug-iranje)

## 3. Zagon
```
cd pot_kjer_so_python_skripte
python validate_two_videos.py pot_do_video_1 pot_do_video_2 pot_do_pravilne_anotacije ime_generirane_json_datoteke
```
Pri tem moramo paziti, da vsem datotekam pripišemo primerne končnice (.mp4 in .json). Video_1 in video_2 morata biti posnetka enakega poskusa (vsak iz svoje kamere). Zaporedje videoposnetkov ni pomembno.

Zadnji argument bo v generirani JSON datoteki podvojen, saj bo pisalo
| "video_name": ime_generirane_json_datoteke.mp4

Primer zagona:
```
python validate_two_videos.py 64210323_video_1.mp4 64210323_video_5.mp4 rocna_anotacija_reformat/64210323_video_1.json nov_json_video_1.json
```
V zgornjem primeru zagona sta oba video posnetka v enaki datoteki kot skripta, ki jo zaganjamo.
Kodo lahko zaženemo tudi z navajanjem absolutne poti, npr.:
```
python validate_two_videos.py "C:\Users\David Zindović\Desktop\Fax-Mag\RV\izziv\izziv main\64210323_video_1.mp4" "C:\Users\David Zindović\Desktop\Fax-Mag\RV\izziv\izziv main\64210323_video_5.mp4" "C:\Users\David Zindović\Desktop\Fax-Mag\RV\izziv\izziv main\rocna_anotacija_reformat\64210323_video_1.json" "64210323_video_1.json"
```