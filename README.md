# warmpool

El Nino and US power demand, from 1870 to the 2026-27 winter

[![ci](https://github.com/adwitiyashukla/warmpool/actions/workflows/ci.yml/badge.svg)](https://github.com/adwitiyashukla/warmpool/actions/workflows/ci.yml)

Live dashboard: https://huggingface.co/spaces/adwitiyashukla/warmpool

On September 10, 2026 NOAA's Climate Prediction Center gave a 75% chance that this El Nino reaches +2.5 C or more on the three month RONI by October to December. That would make it stronger than any El Nino since 1950. By the last week of September the weekly Nino 3.4 anomaly was already +3.2 C. For a utility the real question is what that does to winter load and prices, and how much of it a forecast can actually use.

warmpool works through that question with public data only. It rebuilds an ENSO index back to 1870 and mines 131 years of state climate records for El Nino fingerprints. Then it models how weather drives residential electricity sales in every state, backtests winter forecasts against climatology since 1950, and ends with an outlook for the coming winter. Every number below comes from the real run.

## winter 2026-27

The July to September ENSO index came in at +1.69, the third highest for that time of year in 157 years, behind only 1877 and 1997. Because the index is above +1.0, the forecast weights past winters by how close their ENSO state was (13.2 winters of effective sample size) and turns them into a full distribution for every region. Heating degree days say how cold the winter is. Residential sales go through the load model further down.

| region | heating degree days vs normal | chance colder than normal | residential sales vs normal |
| --- | --- | --- | --- |
| US | -1.7% (80% range -10.8 to +3.3) | 32% | -0.9% (80% range -5.8 to +2.4) |
| ISO New England | -4.2% | 20% | -1.7% |
| NYISO | -3.7% | 20% | -1.8% |
| PJM | -2.6% | 23% | -3.2% |
| MISO North | -3.8% | 10% | -1.7% |
| MISO Central | -4.8% | 29% | -3.0% |
| Northwest | -4.1% | 42% | -1.3% |
| ERCOT | +4.8% | 66% | +2.4% |
| CAISO | +3.1% | 61% | +0.1% |
| Southwest | +6.9% | 74% | +2.5% |

The north leans mild while Texas and the Southwest lean cold, which is the usual El Nino split. The regions are groups of whole states named after the grid that covers most of them, so PJM here means DC, DE, MD, NJ, OH, PA, VA and WV. CPC's historic mark is +2.5, and if the December to February index gets there the US median drops to -5.6% heating degree days. At +3.0 it drops to -8.9%.

## the event

NOAA's RONI series starts in 1950, which leaves only four very strong El Ninos to learn from. warmpool extends it back to 1870 with the PSL HadISST Nino 3.4 series. The two agree closely over the 918 months they share (r = 0.955, rmse 0.24). The older values are calibrated to RONI with a linear fit, and their anomalies are rebuilt with CPC's rule of sliding 30 year base periods. The event catalog then follows NOAA's definition of five straight overlapping seasons at +/-0.5 or beyond.

That gives 75 events since 1870, 40 El Ninos counting the one under way now and 35 La Ninas. Six El Ninos peaked at +2.0 or more.

| winter | peak | source |
| --- | --- | --- |
| 1877-78 | +2.64 | HadISST |
| 1888-89 | +2.10 | HadISST |
| 1982-83 | +2.40 | RONI |
| 1991-92 | +2.12 | RONI |
| 1997-98 | +2.28 | RONI |
| 2015-16 | +2.25 | RONI |

## what El Nino does to US weather

For every state, every month, four variables (temperature, heating and cooling degree days, precipitation) and ENSO leads of 0 to 6 months, warmpool tests the Spearman correlation with the ENSO index over 131 years. That is 16,128 tests, and at that size false hits are guaranteed. Each correlation is judged against 999 phase randomized surrogates (Ebisuzaki), which keep the autocorrelation of the real index. Then the whole field goes through Benjamini-Hochberg at q = 0.10. On their own, 2,672 tests pass p < 0.05. After the correction 965 survive, against about 97 expected false discoveries.

The strongest winter signal is rain in Florida, where February precipitation tracks the index at about r = 0.45 at every lead from 0 to 4 months. January also runs colder along the Gulf when El Nino is on, with r between -0.25 and -0.29 from Texas to Georgia, all significant after the correction.

Clustering the 48 states on their seasonal El Nino responses (Ward linkage, k picked by silhouette, 200 bootstrap resamples for stability) splits the country in two. Thirteen states, from Arizona across the Gulf and up the coast to Delaware, run cool and wet in El Nino winters, and the other 35 run warm or near normal. The split is clear on the map but soft in the numbers, with a silhouette of 0.34 and a bootstrap adjusted Rand index of 0.58.

## past winters like this one

To find years that looked like 2026, warmpool compares this year's January to August ENSO path with every year since 1871 using dynamic time warping, written from scratch with a Sakoe-Chiba band. LB_Keogh lower bounds rule out 146 of the 155 candidate years, so only 9 full DTW runs are needed. The eight closest years are 1902-03, 1997-98, 1972-73, 1965-66, 1963-64, 2023-24, 1925-26 and 1951-52, and they do not agree with each other. US heating degree days in those winters ran from -11.0% (2023-24) to +9.0% (1963-64) of normal.

FP-growth, also written from scratch, mines 131 winters of baskets like {very strong El Nino, mild PJM, wet Southeast}, where mild means the bottom third of heating degree days. Only closed rules are kept, each one is tested with 999 label permutations, and 45 of the 48 closed rules pass Benjamini-Hochberg at q = 0.10. One rule holds in all four very strong El Ninos that have state climate data, 1982-83, 1991-92, 1997-98 and 2015-16. Every one of them was mild in MISO North and Central, PJM, NYISO, the Southeast, SPP, the Northwest and the US as a whole, and wet in the Southeast. That combination shows up in only 5.3% of all winters, so the lift is 18.7 (q = 0.005).

## does any of this beat climatology

Correlations are not forecasts. warmpool backtests five methods on every winter from 1950-51 to 2025-26, each issued from that year's July to September ENSO index, and scores the whole forecast distribution with CRPS. Skill is CRPSS against climatology, so 0 means no better than the long run average and positive means better. The methods are climatology, an ENSO kernel that weights past winters by how close their index was, the DTW analogs, a linear regression on the index, and a gated version that only uses the kernel when the index is beyond +/-1.0.

For US heating degree days nothing beats climatology.

| method | all 76 winters | 17 El Nino winters |
| --- | --- | --- |
| ENSO kernel | -0.021 | -0.043 |
| gated | -0.016 | -0.024 |
| linear | -0.003 | -0.030 |
| analogs | -0.033 | -0.021 |

The national number hides a regional split. In the 17 winters where the index was already +0.5 or more, the kernel beats climatology in 32 of 48 states. The best regions are MISO North at +0.126 (90% bootstrap interval 0.053 to 0.192), the Northwest at +0.118 and the Southwest at +0.093, and the worst are ERCOT at -0.135, MISO South at -0.108 and the Southeast at -0.093. In the 8 winters that started at +1.0 or more, MISO North reaches +0.226. The usable El Nino signal is regional, and it is strongest for northern heating load in El Nino years.

The sales hindcast covers the 31 winters since 1995-96. If the actual winter weather were known, the load model would cut the CRPS of the US sales forecast by 68.5% against climatology (CRPSS 0.685, 90% interval 0.605 to 0.752). Forecasting that weather from the ENSO index gets almost none of it, -0.007 over all 31 winters and +0.051 for the gated method in the 5 El Nino winters.

## from weather to megawatt hours

Residential sales go through one regression for each state, DC, the 14 regions and the US. Each one models log sales per day with year and month effects plus heating and cooling degree days per day, fit on the last 15 years with Newey-West errors. Bills cover parts of two months, so the model also picks how much of the previous month's weather to blend in, out of 0, 0.25, 0.5 and 0.75. It picks 0.25 for 42 of the 64 units. The fits are tight, with a median R squared of 0.978 and the lowest at 0.805 (Maine).

One more heating degree day per day lifts residential sales by 5.6% in Florida, 4.5% in Louisiana and 3.9% in Texas, but only 0.7% in Wisconsin and Minnesota. Most southern homes heat with electricity and most northern homes heat with gas, so the same cold snap moves a Gulf utility's load far more than a Midwest one. For the US as a whole it is 2.6% per heating degree day and 5.1% per cooling degree day.

## prices

A cold winter costs more than the extra megawatt hours. warmpool measures each hub's winter premium, the December to February average price over the September to October average, and regresses it on the heating degree day anomaly of the hub's region. Henry Hub's premium rises 2.24 points for every 1% of extra US heating demand (r = 0.62, p = 0.0006, 27 winters). PJM West power rises 2.55 points (r = 0.80) and Mass Hub 2.84 (r = 0.56). The mild US median for 2026-27 maps to a Henry Hub premium of only +1.2%, inside a wide 80% range of -34.5% to +36.8%, so weather is far from the only thing that moves gas prices.

The Northwest runs on hydro, so a wet winter there should mean cheap Mid-C power the next spring. April's Mid-C minus Palo Verde spread does fall when Idaho, Montana, Oregon and Washington get more precipitation from October to March (r = -0.54, p = 0.011, 25 springs). That single month does not survive the correction across all 18 month and driver tests, though (q = 0.19). The winter ENSO index itself shows no significant link to the spread.

## how it is built

```
download         25 public files from NOAA, EIA and FRED, resumable, sha256 manifest
parse            NOAA fixed width text, two EIA Excel layouts, ICE hub sheets
warehouse        bronze raw files -> silver parquet -> gold DuckDB
quality          20 checks saved with the data, any failed error check stops the run
enso             CPC style anomalies, the HadISST splice, the event catalog
teleconnect      16,128 correlations against surrogate nulls with FDR
cluster          Ward clustering of states with bootstrap stability
dtw, analogs     dynamic time warping with LB_Keogh pruning, k-medoids families
fpgrowth, rules  FP-growth, closed rules and permutation tests
load             weather normalized sales model with a billing lag
backtest         five forecast methods scored with CRPS since 1950
prices           winter premiums and the Mid-C hydro spread
outlook          the 2026-27 forecast and the winter index scenarios
replica          synthetic copies of every raw file with planted effects
views            the dashboard's charts and tables, shared by both front ends
app, space       Streamlit dashboard and its static export for the Space
```

On the real run 19 of the 20 quality checks pass and one warning stays visible, 5 wholesale rows in EIA's files whose delivery ends before it starts. Every tunable lives in config.toml and is checked when it loads. The tests run on a synthetic replica of every raw file, written in the real formats with known effects planted inside, and the pipeline has to find them again. On the full size replica it recovers the planted billing lag in 48 of 49 units and the planted climate regions with an adjusted Rand index of 1.0.

The same raw files were run on Linux in the cloud and on a Windows laptop. All 44 tables match, 25 of them exactly and the rest within 1.4e-10. CI runs ruff and the 123 tests on Linux, macOS and Windows with Python 3.11 to 3.13. The live dashboard is a static export of the Streamlit app, with every chart and table built in Python ahead of time, so it loads fast and never goes to sleep.

## run it

```
git clone https://github.com/adwitiyashukla/warmpool
cd warmpool
python -m venv .venv
.venv\Scripts\pip install -e ".[app,dev]"
.venv\Scripts\warmpool download
.venv\Scripts\warmpool run
.venv\Scripts\streamlit run app\app.py
```

On Linux or Mac use .venv/bin/ in place of .venv\Scripts\ and app/app.py for the app. The download needs no account or key. It pulls 25 files, about 12 MB, and resumes if it gets cut off. On a laptop the download takes about 4 seconds and the full run about 3 minutes.

```
.venv\Scripts\python -m pytest
```

The 123 tests need no network. `warmpool space` writes the static dashboard for the Space to build/space.

## data sources

| source | what warmpool takes | link |
| --- | --- | --- |
| NOAA CPC | RONI, ONI and weekly Nino region temperatures | https://www.cpc.ncep.noaa.gov/data/indices/ |
| NOAA PSL | HadISST Nino 3.4 anomalies back to 1870 | https://psl.noaa.gov/data/timeseries/month/ |
| NOAA NCEI nClimDiv | monthly state temperature, degree days and precipitation since 1895 | https://www.ncei.noaa.gov/pub/data/cirs/climdiv/ |
| EIA-861M | monthly retail electricity sales by state and sector since 1990 | https://www.eia.gov/electricity/data/eia861m/ |
| EIA wholesale markets | daily ICE on-peak power prices at eight hubs since 2001 | https://www.eia.gov/electricity/wholesale/ |
| FRED | daily Henry Hub natural gas spot price since 1997 | https://fred.stlouisfed.org/series/DHHNGSP |

## license

MIT
