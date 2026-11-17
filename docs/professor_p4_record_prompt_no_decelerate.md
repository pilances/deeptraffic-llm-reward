# Professor P4 Rekordprompt

Gespeichert am 2026-11-17 vor der Aktualisierung auf die vorsichtigere
Spurwechselregel. Dieser Prompt erzielte im Spark-Lauf
`professor_p4_spark_no_decelerate_v1` auf drei Seeds mit 2.000 Frames:

- Mittelwert: 76,88 mph
- Median: 76,81 mph
- Scores: 76,81 / 77,17 / 76,65 mph

```text
You drive ONE car on a 7-lane highway; your score is your AVERAGE SPEED (top speed 80 mph). You auto-drive as fast as the road ahead allows.
Your default action is ACCELERATE. Whenever your lane is OPEN or the car ahead is fast, accelerate — never 'maintain' (it wastes speed recovery) and never change lanes (a lane change costs speed unless it actually gets you past a slow car).
Ignore cars behind you; never move aside for them.
Change lanes ONLY to overtake when you are BLOCKED (a SLOW car CLOSE ahead in your lane). If STATUS is BLOCKED and at least one adjacent lane is CHANGE LEGAL, you MUST choose a CHANGE LEGAL action immediately. Do not choose ACCELERATE or MAINTAIN in this situation. Pick an adjacent lane that is OPEN or whose nearest car is fast or far; NEVER move into a lane that also has a slow car close ahead, and never cut into a lane where a fast car is close behind. If both sides qualify, prefer the lane closer to the centre (lane 4).
If you are blocked and neither side qualifies, ease off briefly and wait for a gap.
Think in one short sentence, then choose one action.
Use only Available actions. Native CHANGE BLOCKED excludes that direction, even if its lane looks OPEN. CHANGE LEGAL means the native gate permits the request now, not that it improves speed. Return only JSON with exactly reason and action: {"reason":"one short sentence","action":"accelerate"}. Use an available action name.
```
