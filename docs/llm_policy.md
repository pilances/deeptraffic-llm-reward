# Schritt 3: LLMPolicyAdapter

`src/deeptraffic_llm_reward/llm_policy.py` verbindet die drei vorhandenen
Bausteine. Eingabe ist eine Observation passend zu `DeepTrafficConfig`, Ausgabe
eine Python-Ganzzahl 0..4. Die Funktion führt die Aktion nicht aus.

Der Ablauf eines Aufrufs ist:

1. `prepare_llm_request(observation, config)` erstellt Nachrichten und Hashes.
2. `client.chat(messages)` liefert das vorhandene Request/Response-Format.
3. `resolve_action(response_text, on_invalid=...)` prüft die Aktion.
4. `policy(observation)` gibt die Ganzzahl zurück.

Der Client wird ausdrücklich übergeben. Der Konstruktor selbst ruft weder ein
Modell noch die Simulation auf. Ein echter `OllamaClient` kann später dieselbe
Schnittstelle erfüllen; in diesem Schritt wurde nur ein simulierter Client
verwendet. Jede Entscheidung enthält einen neuen System/User-Prompt ohne
automatisch angehängten Chatverlauf, Wiederholungsversuche oder Aktionscache.

## Selbst ausprobieren, ohne LLM oder Simulation

```python
import numpy as np
from deeptraffic_llm_reward import DeepTrafficConfig
from deeptraffic_llm_reward.llm_policy import LLMPolicyAdapter

class SimulatedClient:
    def chat(self, messages):
        return {
            'request': {'model': 'simulated', 'messages': messages},
            'response': {'done': True, 'message': {'content': '{"action":3}'}},
        }

config = DeepTrafficConfig(lanes_side=1, patches_ahead=2, patches_behind=1)
observation = np.ones(9, dtype=np.float32)
policy = LLMPolicyAdapter(config, SimulatedClient())
action = policy(observation)
print(action)                         # 3 = CHANGE_LEFT
print(policy.last_record['status'])   # ok
print(policy.last_record['resolution'])
```

## Ergebnis und Fehler nachvollziehen

`last_record` enthält nur den letzten Versuch, keine unbegrenzt wachsende Liste.
Enthalten sind Konfiguration, Fehlerregel, vorbereitete Nachrichten/Hashes,
Client-Anfrage und -Antwort, Parsergebnis und gesamte Laufzeit in Sekunden.
Die Kopien der Nachrichten/Antwort verhindern, dass nachträgliche Änderungen
des Clients die gespeicherten Daten verändern. Das Dictionary ist für spätere
JSON-Protokolle geeignet. Es werden automatisch keine Dateien erzeugt.

Statuswerte: `ok`, `fallback` oder `error`; `stage` zeigt `prepare`, `chat` oder
`parse`. Bei einem Fehler werden zusätzlich Typ und Grund festgehalten und die
Exception weitergereicht. Vor jedem Aufruf wird der alte Datensatz ersetzt.
Bei fehlgeschlagenen Aufrufen ist `resolution=None`, statt eine alte Aktion
als vermeintlich neues Ergebnis auszuweisen.

Standard ist `on_invalid='raise'`. Mit `on_invalid='maintain'` gilt der in
[llm_action.md](llm_action.md) beschriebene, explizit markierte Fallback für
Parserfehler. Kommunikations- und Programmierfehler werden nicht in Aktionen
umgewandelt. Der Client ist weiterhin für das Prüfen seines Response-Umschlags
zuständig. Eine Adapterinstanz ist für sequenzielle Aufrufe ausgelegt.

Der Adapter ist bereit für einen späteren Live-Test, wurde aber noch nicht mit
einem echten Modell oder `env.step()` verbunden. Es wurden keine Fahrqualität,
Benchmarkwerte oder echte Inferenzlatenzen gemessen.

## Verifikation

```sh
.venv/bin/python -m pytest tests/test_llm_policy.py tests/test_llm_action.py tests/test_llm_client.py tests/test_llm_state.py -ra
```

58 Tests bestanden, davon 11 neue Adaptertests. Geprüft werden alle fünf Aktionen,
Nachrichtenübergabe und volle Rastergröße, unveränderte Eingaben, unabhängige
Aufrufe ohne Historie, Protokolldaten, Parserabbruch, expliziter Fallback und
Kommunikationsfehler ohne Retry. In den Adaptertests sind echte HTTP-Aufrufe
sowie `DeepTrafficEnv.reset/step` zusätzlich als verbotene Aufrufe abgesichert.
