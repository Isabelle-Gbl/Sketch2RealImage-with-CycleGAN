import os
import json
import argparse
from PIL import Image, ImageDraw, ImageOps, ImageEnhance

def lade_zeichnungsdaten(pfad):
    """Lädt Zeichnungen aus einer .ndjson-Datei."""
    if not os.path.exists(pfad):
        raise FileNotFoundError(f"Die Datei {pfad} wurde nicht gefunden.")
    with open(pfad, 'r') as f:
        return [json.loads(zeile) for zeile in f]

def zeichnung_zu_bild(drawing, bildgroesse=(256, 256)):
    """Erstellt ein zentriertes Bild aus einer Zeichnung."""
    bild = Image.new("RGB", bildgroesse, "white")
    zeichnung = ImageDraw.Draw(bild)
    
    # Bestimme die Bounding Box der Zeichnung
    min_x = min(min(pfad[0]) for pfad in drawing)
    max_x = max(max(pfad[0]) for pfad in drawing)
    min_y = min(min(pfad[1]) for pfad in drawing)
    max_y = max(max(pfad[1]) for pfad in drawing)
    
    zeichnung_width = max_x - min_x
    zeichnung_height = max_y - min_y
    
    # Skaliere die Zeichnung auf die maximale Größe, ohne Verzerrung
    scale = min((bildgroesse[0] - 10) / zeichnung_width, (bildgroesse[1] - 10) / zeichnung_height)
    
    # Berechne den Mittelpunkt der Zeichnung
    x_mittelpunkt = (max_x + min_x) / 2
    y_mittelpunkt = (max_y + min_y) / 2
    
    for pfad in drawing:
        x_werte = [((x - x_mittelpunkt) * scale + bildgroesse[0] / 2) for x in pfad[0]]
        y_werte = [((y - y_mittelpunkt) * scale + bildgroesse[1] / 2) for y in pfad[1]]
        koordinaten = list(zip(x_werte, y_werte))
        zeichnung.line(koordinaten, fill="black", width=2)
    
    # Zentriere die Zeichnung
    bild = passe_bildgroesse_an(bild, bildgroesse)
    return bild

def passe_bildgroesse_an(bild, zielgroesse=(256, 256)):
    """Skaliert die Zeichnung größtmöglich auf 256x256 ohne Verzerrung und zentriert sie."""
    bild.thumbnail(zielgroesse, Image.Resampling.LANCZOS)
    neues_bild = Image.new("RGB", zielgroesse, "white")
    x_offset = (zielgroesse[0] - bild.size[0]) // 2
    y_offset = (zielgroesse[1] - bild.size[1]) // 2
    neues_bild.paste(bild, (x_offset, y_offset))
    return neues_bild

def konvertiere_in_schwarz_weiss(bild, schwellenwert=128):
    """Konvertiert das Bild in Schwarz-Weiß."""
    graustufen_bild = bild.convert("L")
    return graustufen_bild.point(lambda p: 255 if p > schwellenwert else 0, mode="1")

def erhoehe_kontrast(bild, faktor=2):
    """Erhöht den Kontrast des Bildes."""
    enhancer = ImageEnhance.Contrast(bild.convert("L"))
    return enhancer.enhance(faktor)

def speichere_zeichnungsbilder(daten, zielordner, anzahl_bilder, bildgroesse=(256, 256), schwellenwert=128, kontrastfaktor=2):
    """Speichert die Zeichnungen als Bilder."""
    os.makedirs(zielordner, exist_ok=True)
    for i, eintrag in enumerate(daten[:anzahl_bilder]):
        if "drawing" in eintrag:
            dateiname = os.path.join(zielordner, f"bild_{i}.png")
            bild = zeichnung_zu_bild(eintrag["drawing"], bildgroesse)
            bild = konvertiere_in_schwarz_weiss(bild, schwellenwert)
            bild = erhoehe_kontrast(bild, kontrastfaktor)
            bild.save(dateiname)

def verarbeite_ndjson(motiv_name, image_count, root_path):
    """Verarbeitet eine .ndjson-Datei und speichert die Bilder."""
    ndjson_ordner = os.path.join(root_path, motiv_name)
    ndjson_dateien = [f for f in os.listdir(ndjson_ordner) if f.endswith(".ndjson")]
    
    if not ndjson_dateien:
        print(f"Keine .ndjson-Datei für {motiv_name} gefunden.")
        return
    
    ndjson_pfad = os.path.join(ndjson_ordner, ndjson_dateien[0])
    zielordner = os.path.join(root_path, motiv_name, "trainA")
    
    try:
        daten = lade_zeichnungsdaten(ndjson_pfad)
    except FileNotFoundError as e:
        print(e)
        return
    
    speichere_zeichnungsbilder(daten, zielordner, image_count)
    print(f"Bilder für {motiv_name} wurden in {zielordner} gespeichert.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Konvertiert Zeichnungen aus einer .ndjson-Datei in Bilder für CycleGAN.")
    parser.add_argument("motiv_name", type=str, help="Name des Motiv-Ordners")
    parser.add_argument("image_count", type=int, help="Anzahl der zu verarbeitenden Bilder")
    parser.add_argument("--root_path", type=str, default="Orginal_CycleGAN_Repository/datasets", help="Pfad zum Hauptverzeichnis der Daten")

    args = parser.parse_args()
    verarbeite_ndjson(args.motiv_name, args.image_count, args.root_path)
