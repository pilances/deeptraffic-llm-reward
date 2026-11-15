Die Live-Integration lässt sich mit `.venv/bin/python scripts/run_llm_integration.py` ausführen.
Ein neuer Ausgabeordner kann über `--out` gewählt werden; vorhandene Ordner werden nicht überschrieben.

Das Skript prüft zuerst einen synthetischen State, dann genau einen Gymnasium-Schritt
und anschließend zwei vollständige, jeweils sechs Frames lange Episoden (Seeds 42 und 43).
Die Konfiguration verwendet elf kontrollierte Fahrzeuge und einen State mit 52 × 13 Zellen.
Diese kurzen Episoden dienen ausschließlich dem technischen Funktionstest. Ihre Geschwindigkeit
ist kein Vergleichswert zur historischen Benchmark.

`decisions.jsonl` enthält pro Anfrage State, Prompt, Antwort, Parserentscheidung, Kontext
und Laufzeit. `report.json` enthält Konfiguration, Modellmetadaten, ausgeführte Aktionen,
Rewards und resultierende States. Beide Dateien entstehen durch das Skript und können
für einen neuen Test in einem anderen Ausgabeordner erneut erzeugt werden.
Parser- und Kommunikationsfehler brechen den Lauf ab; es werden keine Ersatzaktionen verwendet.
Die Umgebung wartet während der Anfrage auf die Aktion. Der Client entlädt das Modell
nach jeder Anfrage gemäß seiner bisherigen Einstellung, wodurch Ladezeit wiederholt anfällt.
