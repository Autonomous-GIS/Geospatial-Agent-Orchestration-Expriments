# Analytical Specification Case Study

This folder preserves the evidence for the three orchestrated cases in Section 6.
Each case contains a copy of its workflow and the identified agent-produced artifacts.
The original files outside this repository were not modified.

## Case B Fully Specified

`Case_B_Fully_Specified` contains the fully specified workflow, its calculated index table, joined county layer, projected GeoPackage, and map. The copied map is byte-identical to Figure 3b.

The workflow identifies two upstream retrieval outputs (`download_the_386195.csv` and `download_the_599189.gpkg`). Those exact files were not found in the candidate source locations, so they are not represented as copied artifacts.

## Case C Indicator Specified

`Case_C_Indicator_Specified` contains the indicator-specified workflow, projected county layer, calculated economic-distress GeoJSON, and map. The copied map is byte-identical to Figure 3c.

The workflow identifies two upstream retrieval outputs (`download_the_622091.csv` and `download_the_968200.gpkg`). Those exact files were not found in the candidate source locations, so they are not represented as copied artifacts.

## Case D Goal Oriented

`Case_D_Goal_Oriented` contains the goal-oriented workflow and all identified discovery, vector-GIS, and cartography outputs. Two successful cartography outputs were retained: `economic_distress_index_map_initial.png` is the first map output, and `economic_distress_index_map_final_figure_3d.png` is byte-identical to Figure 3d.

## Source locations

- Cases B and C: `/Users/alikhosravikazazi/Desktop/Geospatial-Agentic-Services-main new`
- Case D: `/Users/alikhosravikazazi/Desktop/GAS_Final`

## Public release note

The public repository preserves workflow structure, execution evidence, and analytical artifacts. Any embedded credential values are redacted for publication; this does not alter workflow logic, task content, or analytical outputs.
