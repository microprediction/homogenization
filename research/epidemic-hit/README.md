# Early growth against herd immunity: can fast-switching transmission explain the gap?

## Question
SIR/SEIR models fitted to early epidemic growth imply a reproduction number R and a herd-immunity threshold
1 - 1/R. For COVID-19 the first waves turned over at much lower immunity than those thresholds. Can transmission
that switches between regimes, in many places at once, produce this gap, with a Green-Kubo size that can be computed?

Origin: Cotton (27 May 2020), "A Fundamental Theorem for Epidemiology" (LinkedIn,
https://www.linkedin.com/pulse/fundamental-theorem-epidemiology-peter-cotton-phd/): states the paradox (doubling times
under a week, yet peak infection at about 20% penetration), argues via de Finetti that only the orbit of symmetric
models matters, and gives the susceptible fraction at herd immunity as the usual answer divided by the harmonic
mean of the transmission multiplier. Then Cotton (2020), "Addressing the Herd Immunity Paradox Using Symmetry, Convexity Adjustments and Bond Prices",
arXiv:2006.07341 (June 2020). It poses exactly this question and answers it for a mixture of independent
sub-populations: a growth convexity G (moment generating function of the transmission rate; Vasicek's bond formula
for an OU rate) raises mean early growth, a harmonic-mean convexity J raises the susceptible fraction at the peak,
and s(t*) = 1/(F J G R0). Also Cotton (2020), "Repeat Contacts and the Spread of Disease", arXiv:2005.10311 (May 2020):
staleness of acquaintances; and the LinkedIn pieces "Got Milk?" and "How population shape tilts your odds of getting
COVID-19". All predate Tkachenko et al. (arXiv 2008.08142, August 2020).

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
