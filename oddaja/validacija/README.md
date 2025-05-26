# Navodila za uporabo validacijske skripte

## 1. Kam dati svojo kodo
V datoteki `validate_single_video.py` je označen prostor za vašo kodo:
```python
"""
YOUR CODE HERE. Tukaj vključite klicanje vaše funkcije, ki obdela video in vrne/shrani json z rezultati.
"""
```
Tukaj vključite klic vaše funkcije, ki obdela video in ustvari JSON z rezultati.

## 2. Format vašega output JSONa
Vaš algoritem mora ustvariti JSON datoteko v naslednjem formatu:
```json
{
    "1": ["prazna_roka"],
    "2": ["prazna_roka"],
    "3": ["prijemanje_pina"],
    ...
}
```
Kjer:
- Ključi so številke okvirjev (frames) v videu
- Vrednosti so seznami dogodkov za ta okvir (v našem primeru bo vedno samo en dogodek)

## 3. Kako poklicati skripto
Iz terminala pokličite:
```bash
python validate_single_video.py <pot_do_videa> <pot_do_ground_truth_json> <pot_do_student_output_json>
```

Primer:
```bash
python validate_single_video.py primer.mp4 primer.json primer_output.json
```

## 4. Kaj bo skripta naredila
- Preveri vaš output JSON
- Izpiše metrike (točnost, število pravilnih/napačnih napovedi)
- Ustvari confusion matrix (shrani se kot PNG datoteka)

## 5. Pričakovani dogodki
Vaš algoritem mora razpoznavati naslednje dogodke:
- prazna_roka
- prijemanje_pina
- prenos_pina
- odlaganje_pina 