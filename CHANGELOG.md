# Versionsverlauf – MusicAI

Dieses Dokument erfasst Änderungen, die Verhalten, Daten oder Voraussetzungen des Projekts betreffen.

## 0.2.4 – 19. Oktober 2026

- Ausschließlich lokale NAS-Pfade: UNC-/Proxy-Pfade werden auf den vorhandenen gemappten NAS-Laufwerksbuchstaben (hier `B:\`) umgestellt. SQLite verwendet damit keinen UNC-Zugriff mehr.
- Alte Proxy-/Containerpfade werden beim Start auf den lokalen NAS-Projektpfad umgestellt; doppelte Creator-Verknüpfungen desselben Ordners werden in der Datenbank sicher zusammengeführt.
- Der Starter verhindert parallele MusicAI-Instanzen mit einer NAS-kompatiblen Sperre.
- SQLite hält pro MusicAI-Prozess eine serialisierte Verbindung statt bei jeder Abfrage neue SMB-Dateisperren zu erzeugen.

## 0.2.1 – 18. Oktober 2026

- Creator-Ordner werden im Dashboard über einen NAS-Ordnerbrowser ausgewählt; das manuelle Pfadfeld wurde entfernt.
- Der Server beschränkt die Ordnernavigation und jede verknüpfte Quelle auf die NAS-Freigabe des Projekts. Lokale Laufwerke außerhalb der Freigabe werden abgewiesen.
- SQLite-Zugriffe auf der NAS-Freigabe wiederholen vorübergehende SMB-I/O- und Sperrfehler; ein zweiter Serverstart startet keinen zusätzlichen Watcher mehr.
- Der Starter hält für den Dashboard-Betrieb eine NAS-kompatible Ein-Instanz-Sperre und verweigert parallele Starts eindeutig.

## 0.2.0 – 17. Oktober 2026

- Training prüft die Modell- und Quantisierungsvoraussetzungen vor dem zeitintensiven Sammeln von Audioausschnitten.
- Fehlendes `bitsandbytes` wird mit einer präzisen, direkt ausführbaren Lösung gemeldet.
- Neues strukturiertes Fehlerdiagnose-Protokoll: `logs/diagnostics.log` enthält Start-, Job- und Fehlerereignisse als JSON-Zeilen.
- Fehleranalyse erkennt Probleme mit `bitsandbytes` und `decoder_start_token_id` gezielt.
- `python start.py version` zeigt Version und Pfad zu diesem Verlauf; `doctor` enthält die Version.

## 0.1.0 – 12. Oktober 2026

- Erste lokale MusicGen-/LoRA-Pipeline mit Analyse, Stilprofilen, Rechte-Sperre, WebUI und Hintergrundjobs.
