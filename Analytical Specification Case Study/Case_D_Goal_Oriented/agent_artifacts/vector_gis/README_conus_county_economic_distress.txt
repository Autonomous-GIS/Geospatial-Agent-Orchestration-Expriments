County Economic Distress Index — ACS 2017–2021 5-year estimates

For each county, standardize each supplied adverse ACS proportion across all 3,108 counties using population standard deviation (z=(x-mean)/SD); EDI z mean is the unweighted arithmetic mean of the six component z-scores; edi_score = 50 + 10*edi_z_mean. Higher values indicate greater relative economic distress.

Inputs: pct_less_than_30k, pct_less_than_high_school, pct_below_poverty, pct_unemployment, pct_with_snap_public_assistance, and pct_with_ssi. These are supplied derived proportions in the source GeoPackage; raw ACS fields and geographic identifiers are retained.

Coverage: 3,108 county-equivalent features for the 48 contiguous states plus DC. Alaska, Hawaii, and Puerto Rico are excluded.
Geometry: source EPSG:4269/NAD83; output EPSG:5070/NAD83 / Conus Albers. Layer: county_edi.

The score is relative to this exact county universe and vintage. It should not be interpreted as a causal measure, an absolute hardship threshold, or a percentage. edi_percentile is included as an optional rank-based display field (0–100).
