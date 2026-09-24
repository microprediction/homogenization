# Early growth against herd immunity: can fast-switching transmission explain the gap?

## Question
SIR/SEIR models fitted to early epidemic growth imply a reproduction number R and a herd-immunity threshold
1 - 1/R. For COVID-19 the first waves turned over at much lower immunity than those thresholds. Can transmission
that switches between regimes, in many places at once, produce this gap, with a Green-Kubo size that can be computed?

Origin: Cotton (2020), "Got Milk? On Homogenization and Survival Probability" (LinkedIn): transmission probability and
the number infected "are related by fluctuations across space and time". The persistent-spatial version is Cotton
(2020), "How population shape tilts your odds of getting COVID-19" (Jensen / harmonic-mean correction to the HIT).

## Layout
- fetch_data.py: NYT US county cases and deaths for 2020 into data/ (gitignored).
- analysis/01_communities.py: simulation of many communities with independently switching transmission.
- analysis/02_county_growth.py, 03_compounding.py: county growth heterogeneity, persistence, compounding.
- FINDINGS.md: dated log.

## Prior art to position against
- Britton, Ball, Trapman (2020), Science 369:846-849, doi:10.1126/science.abc6810: persistent heterogeneity (age,
  activity) lowers the disease-induced herd immunity level.
- Gomes et al. (2022), J. Theor. Biol. 540:111063, doi:10.1016/j.jtbi.2022.111063: variation in susceptibility or
  exposure lowers the HIT.
- Tkachenko et al. (2021), PNAS 118:e2015972118, doi:10.1073/pnas.2015972118: time-dependent (switching)
  individual activity gives transient suppression, not herd immunity. The closest prior work to the switching story.
