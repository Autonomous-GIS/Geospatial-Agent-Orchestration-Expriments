# Case D External Data Availability

The two GeoPackages below exceed GitHub's per-file size limit. They are public downloads and preserve the full Section 6 goal-oriented case evidence. Download each file into the local path listed below, then verify its SHA-256 checksum.

| File | Purpose | Public download | Expected local path | SHA-256 |
| --- | --- | --- | --- | --- |
| `conus_county_economic_distress_2017_2021.gpkg` | Final projected county Economic Distress Index dataset used to create Figure 3d. | [Google Drive](https://drive.google.com/file/d/1MoTuDZqVyKvWSDjaFOy35SD5f1aeATf3/view?usp=sharing) | `agent_artifacts/vector_gis/conus_county_economic_distress_2017_2021.gpkg` | `d7e72de8df7cb2d82c5e2f8a9a99698a60cf498f8954fe4bd75fb4ad63c3457f` |
| `county_edi_2017_2021.gpkg` | Data-discovery output with county geometry and ACS 2017–2021 indicator inputs used by the Case D workflow. | [Google Drive](https://drive.google.com/file/d/17B7YSISJLrcx3Tb5VFKDsM8CkBccDk1i/view?usp=sharing) | `agent_artifacts/data_discovery/county_edi_2017_2021.gpkg` | `26ee4d056b6e5a3b27d2721abcdb24bb7c0a7ba2e01ccb7ba2cfbf2a266c1a1b` |

On macOS or Linux, verify a downloaded file with:

```sh
shasum -a 256 path/to/file.gpkg
```
