# Premier League Player Impact Ratings

**Who actually moves the needle when they're on the pitch?** This project ranks
Premier League players (2023/24 – 2025/26) by their on-pitch impact, built up in
stages: the intuitive **on/off goal plus-minus**, its failure modes, the fix —
**regularized adjusted plus-minus (RAPM)** on match *stints* with a non-penalty
xG response and game-state controls — and an **empirical-Bayes finishing
overlay** so clinical finishers are credited for their own conversion. The
headline metric is `impact90 = npxG-RAPM + finishing per 90`, validated on a
held-out set of future matches.

![Top 20 players by xG-RAPM](outputs/top20_xg_rapm.png)

## The idea

Goal plus-minus asks a simple question: *how does a team's goal difference per 90
change when a player is on the pitch versus off it?* Simple — and, on its own,
badly misleading in soccer:

1. **Small samples.** Goals are rare (~2.8/match). A few lucky stints make a
   squad player look elite.
2. **Collinearity.** Regular starters share 70%+ of their minutes; on/off numbers
   describe the *team*, not the player.
3. **Substitution bias.** Subs enter in non-random game states (chasing,
   protecting, garbage time).

The fix, borrowed from NBA analytics and applied to soccer in the academic
literature (Sæbø & Hvattum 2015; Kharrat, López Peña & Boukas 2020), is to regress
**stint outcomes on everyone on the pitch simultaneously**, with ridge
regularization to tame the collinearity.

## Method

**Stints.** Each match is segmented into maximal intervals with an unchanged set
of players *and an unchanged score* — boundaries at substitutions, red cards, and
goals (10,315 stints across 1,140 matches). Every stint records both rosters, its
duration, the score when it began, and the goals/xG created by each side inside it.

**The regression.** One row per stint (~7,600 of them):

- **Response:** stint goal differential (home − away), scaled to per-90; rows
  weighted by stint duration. The headline variant uses **non-penalty xG** — a
  penalty is ~0.76 xG awarded to the whole lineup for an individual event, so
  spot kicks are excluded from the collective model and handled by the finishing
  overlay.
- **Player columns:** +1 if on the pitch for the home side, −1 for the away side.
  Players under 900 career minutes pool into one *replacement* column, so
  coefficients read as **impact per 90 over a replacement-level player**.
- **Controls:** an unpenalized intercept (home advantage: recovered at
  +0.28 xG/90), a man-count differential (an extra man is worth ≈ 1.0 xG/90),
  score-state dummies, and match-phase dummies. The fitted game-state terms
  recover known dynamics — teams leading by one generate ≈ 0.13 xG/90 less
  (sitting back), trailing teams ≈ 0.12 more (pushing).
- **Ridge λ** chosen by cross-validation with folds **grouped by match**, so no
  match straddles the train/test split.
- **Uncertainty** from a cluster bootstrap (resampling matches, 500 draws).

**Why xG — and what about clinical finishers?** Goals are a noisy readout of
match control; chance quality carries more signal per minute. But a pure-xG
model rates a wasteful striker level with a clinical one, and switching the
whole regression to goals doesn't fix that — a goal credits all 11 players, so
finishing skill smears across the lineup. Finishing is an *individual* skill,
so it gets an individual treatment: each player's own goals-minus-xG per shot,
**shrunk with an empirical-Bayes prior** (τ ≈ 0.010 goals/shot, estimated from
the data — finishing skill is real but small, and raw hot streaks are mostly
luck). The shrunk edge times the player's shot volume gives `finishing_per90`,
and the headline is `impact90 = npxG-RAPM + finishing_per90`. Face validity:
the overlay's biggest losers are famously wasteful profiles (Darwin Núñez,
Calvert-Lewin); its winners (Cunha, Foden) are known clinical finishers.

**Out-of-sample check** ([notebook 05](notebooks/05_validation_and_final_rankings.ipynb)):
trained on matches before March 2026 and predicting the final ~10 gameweeks from
kickoff XIs alone, lineup ratings rank match npxG differentials better than a
team-strength baseline (Spearman 0.33 vs 0.28) and beat home-advantage-only on
error — but do not improve MSE over team strength; single matches are noisy and
the writeup says so.

![What adjustment does](outputs/naive_vs_rapm.png)

## Validation

- **Hard reconciliation gates** on every match: stint goals sum exactly to the
  final score; every match starts 11v11 and player counts only ever fall; each
  player's summed stint minutes equal their appearance interval; final scores
  match an independent source (football-data.co.uk).
- **Split-half reliability:** the model is fit on two disjoint halves of the
  match sample and the two rating vectors are correlated.
- **Predictive holdout:** trained before March 2026, kickoff-XI predictions of
  the final ~10 gameweeks are scored against team-strength and home-advantage
  baselines (results reported honestly in notebook 05).
- **Sanity recoveries:** home advantage and the value of a man advantage are
  estimated, not assumed — both land where the literature says they should.
- **Unit tests** cover the stint builder's edge cases: same-minute double subs,
  red cards, sub-of-sub chains, stoppage-time subs that extend the match clock.

## Results

Top 10 by `impact90` (npxG-RAPM + finishing per 90 over a replacement-level
player), 2023/24–2025/26, with 90% bootstrap intervals:

| # | Player | Team | Pos | Minutes | Impact /90 | 90% CI |
|---|--------|------|-----|--------:|-----------:|--------|
| 1 | Bruno Guimarães | Newcastle United | MID | 9,133 | **+0.34** | +0.20 … +0.44 |
| 2 | Rodri | Manchester City | MID | 4,564 | **+0.24** | +0.13 … +0.36 |
| 3 | Kevin Schade | Brentford | FWD | 5,445 | **+0.24** | +0.11 … +0.36 |
| 4 | Jacob Murphy | Newcastle United | FWD | 5,314 | **+0.24** | +0.10 … +0.38 |
| 5 | Trent Alexander-Arnold | Liverpool | DEF | 4,617 | **+0.22** | +0.11 … +0.35 |
| 6 | William Saliba | Arsenal | DEF | 9,139 | **+0.21** | +0.13 … +0.29 |
| 7 | Rúben Dias | Manchester City | DEF | 7,012 | **+0.21** | +0.10 … +0.33 |
| 8 | Callum Wilson | West Ham | FWD | 2,537 | **+0.20** | +0.05 … +0.34 |
| 9 | Kaoru Mitoma | Brighton | FWD | 5,854 | **+0.19** | +0.07 … +0.31 |
| 10 | Beto | Everton | FWD | 4,008 | **+0.19** | +0.05 … +0.31 |

The model finds the names an informed fan would expect (Rodri's famous on/off
effect survives full adjustment; Saliba and Dias anchor the two best defenses)
alongside genuinely interesting cases — Bruno Guimarães tops the board on
enormous minutes, and role players like Kevin Schade and Jacob Murphy post
strong per-90 impact that raw plus-minus hides. Every interval is wide: three
seasons of stints identifies individual impact only coarsely, and the chart
says so honestly.

Full rankings live in [outputs/rankings.csv](outputs/rankings.csv); the
notebooks tell the full story:

| Notebook | What it shows |
|---|---|
| [01 — Data & EDA](notebooks/01_data_and_eda.ipynb) | The stint table and its reconciliation gates |
| [02 — Naive plus-minus](notebooks/02_naive_plusminus.ipynb) | The baseline and its three failure modes, demonstrated |
| [03 — RAPM](notebooks/03_rapm.ipynb) | The adjusted model on goals |
| [04 — xG-RAPM](notebooks/04_xg_rapm.ipynb) | The xG response and method comparison |
| [05 — Validation & rankings](notebooks/05_validation_and_final_rankings.ipynb) | Reliability, uncertainty, final tables |

![Impact vs minutes](outputs/rating_vs_minutes.png)

## Data

All free sources. **[Understat](https://understat.com)** supplies everything the
model needs from a single, internally consistent source: match rosters with
substitution linkage (each sub's record points to the player they replaced, whose
`time` field is the sub minute — verified exact against real matches), red-card
exits, and shot-level xG with minutes. Final scores are cross-checked against
**[football-data.co.uk](https://www.football-data.co.uk)** CSVs.

> The original plan was FBref match reports via `soccerdata`; FBref now
> hard-blocks scraping (HTTP 403 even with browser-impersonation TLS), which is
> itself a useful data-engineering lesson: design the pipeline so the source is
> swappable and every raw response is cached.

Known approximations, accepted and documented: Understat's clock caps regulation
at 90 (stoppage-time subs chain past it), shot minutes are integers, and a shot
at an exact stint boundary is attributed to the later stint (±1 min ambiguity).

## Reproduce it

```bash
uv sync                 # Python 3.12 environment
make all                # scrape (cached, resumable) -> build -> test -> model -> charts
make notebooks          # execute the narrative notebooks in place
```

`plimpact` is also a CLI: `uv run plimpact scrape | build | model | outputs`.

```
src/plimpact/
├── scrape.py     # cached, resumable Understat + football-data pulls
├── parse.py      # raw JSON -> typed appearance/shot records
├── stints.py     # match -> stint segmentation (the core data structure)
├── naive.py      # on/off plus-minus baseline
├── rapm.py       # sparse design matrix, ridge, grouped CV, bootstrap
├── finishing.py  # empirical-Bayes finishing overlay
├── predict.py    # out-of-sample holdout validation
├── model.py      # orchestration: naive -> RAPM -> npxG-RAPM -> impact90
├── validate.py   # reconciliation gates
└── viz.py        # README charts
```

## References

- Sæbø, O.D. & Hvattum, L.M. (2015). *Evaluating the efficiency of the
  association football transfer market using regression-based player ratings.*
- Kharrat, T., López Peña, J. & Boukas, I. (2020). *Plus-minus player ratings for
  soccer.* Annals of Operations Research.
- Sill, J. (2010). *Improved NBA adjusted plus-minus using regularization and
  out-of-sample testing.* MIT Sloan Sports Analytics Conference.
