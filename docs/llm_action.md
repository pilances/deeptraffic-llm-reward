# Schritt 1 und 2: Antwortparser und Fallback

Die neue Datei `src/deeptraffic_llm_reward/llm_action.py` verarbeitet ausschließlich
Antworttexte. Sie ruft weder Ollama noch `env.step()` auf. Der bestehende
Kommunikationsclient bleibt unverändert. Der anschließend implementierte
Policy-Adapter ist in [llm_policy.md](llm_policy.md) beschrieben.

## Schritt 1: Text prüfen

```python
from deeptraffic_llm_reward.llm_action import parse_action

action = parse_action('{"action":3}')
print(action)  # 3 = CHANGE_LEFT
```

`parse_action(text)` liest genau ein JSON-Objekt mit genau einem Feld `action`.
Der Wert muss eine Ganzzahl aus dem Action-Enum der Umgebung sein:
0=MAINTAIN, 1=ACCELERATE, 2=DECELERATE, 3=CHANGE_LEFT, 4=CHANGE_RIGHT.
Äußerer Leerraum ist erlaubt. Zusätzliche Felder, Markdown, Freitext,
abgeschnittene Antworten, mehrere JSON-Objekte, doppelte Schlüssel, Strings,
Boolesche Werte, Fließkommazahlen und Werte außerhalb 0..4 werden abgelehnt.
Der Parser repariert oder konvertiert keine Antworten.

Bei ungültigem Text entsteht `ActionParseError` (eine Unterklasse von
`ValueError`). Das Attribut `reason` nennt einen stabilen Fehlercode:
`invalid_json`, `invalid_schema`, `duplicate_key`, `invalid_action_type` oder
`invalid_action_value`. Nichttext-Eingaben sind Aufruferfehler (`TypeError`).

## Schritt 2: Verhalten bei Fehlern wählen

```python
from deeptraffic_llm_reward.llm_action import resolve_action

valid = resolve_action('{"action":3}')  # Standard: Fehler weiterreichen
print(valid.action)                    # 3

fallback = resolve_action('keine gültige Antwort', on_invalid='maintain')
print(fallback.to_dict())
# {
#   'raw_response': 'keine gültige Antwort',
#   'action': 0,
#   'parsed_action': None,
#   'fallback_used': True,
#   'error_reason': 'invalid_json',
#   'on_invalid': 'maintain'
# }
```

`resolve_action` ruft den Parser auf und liefert ein unveränderliches
`ActionResolution`-Ergebnis. `on_invalid='raise'` ist der Standard: Es wird keine
Ersatzaktion gewählt. Nur mit `on_invalid='maintain'` werden Parserfehler auf
MAINTAIN abgebildet. Diese Option erhält Rohantwort und Fehlergrund und gibt
zusätzlich eine Python-Logging-Warnung aus. Deren `action_resolution`-Attribut
enthält denselben strukturierten Datensatz. Der spätere Adapter kann mit
`to_dict()` auch alle gültigen Entscheidungen als JSON protokollieren.

Eine gültige Antwort `{"action":0}` hat `parsed_action=0` und
`fallback_used=False`. Ein Fallback hat `parsed_action=None` und
`fallback_used=True`. Die Bewertungslogik kann dadurch die Fehlerrate separat
ausweisen; MAINTAIN ist keine Garantie für sicheres oder gutes Fahrverhalten.
Für einen späteren Benchmark muss die Fehlerregel vorab festgelegt werden.

Kommunikationsfehler, Timeouts und andere Programmierfehler werden hier nicht
abgefangen. Die Funktion selbst schreibt keine Dateien. Logging-Ausgabe und
Dateispeicherung werden vom Aufrufer konfiguriert. Keine neue Abhängigkeit oder
generierte Datei ist für die Nutzung erforderlich.

## Ausgeführte Tests

```sh
.venv/bin/python -m pytest tests/test_llm_action.py tests/test_llm_client.py tests/test_llm_state.py -ra
```

Die Tests prüfen alle fünf Aktionen, ungültige Antwortklassen, Standardabbruch,
expliziten Fallback, JSON-serialisierbare Protokolldaten und die strukturierte
Warnung. Sämtliche Tests laufen offline, ohne Inferenz oder Aktionsausführung.
