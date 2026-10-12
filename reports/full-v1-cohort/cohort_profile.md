# Aggregate cohort characteristics

Descriptive source profile; no model performance or individual records are included.

Patient records: **40,336**. Recorded hours: **1,552,210**.
Parquet SHA256: `298a177fd3de8cee4e270a71d0a371fc58588316fdbebeb7bdaa551d971d284a`.

Full versus reduced cohort status must be taken from the accompanying run manifest.

## Hospital-level population

| Hospital | Records | Recorded hours | Ever-positive records | Prevalence | Stay minimum | Stay median [Q1, Q3] | Stay maximum |
|---|---:|---:|---:|---:|---:|---|---:|
|A|20,336|790,215|1,790|8.80%|8|39 [25, 47]|336|
|B|20,000|761,995|1,142|5.71%|8|38 [23, 47]|336|

Stay duration is recorded rows, not total hospital duration. Ever-positive labels differ from eligible pre-onset events.

## Patient-weighted demographics

Age uses the first nonmissing value per record. Source Age=100 denotes the >=90 category; it is not an exact age. Recorded Gender codes do not establish gender identity.

### Hospital A

Age median [Q1, Q3]: 64.67 [52.29, 75.66]; missing: 0.

| Age category | Records |
|---|---:|
|<18|0|
|18–39|2,032|
|40–64|8,284|
|65–79|6,975|
|80–89|3,045|
|>=90 coded 100|0|
|>=90 other coding|0|
|Missing|0|

| Recorded Gender | Records |
|---|---:|
|Code 0|8,502|
|Code 1|11,834|
|Missing|0|
|Other code|0|

### Hospital A: predictor availability and distributions

Missingness always uses hourly rows. Distribution counts/quantiles are hour-weighted except the five demographic/administrative fields marked patient. ICULOS is hourly.

| Predictor | Missing hours | Missing fraction | Distribution unit | Observations | Median | Q1 | Q3 | IQR |
|---|---:|---:|---|---:|---:|---:|---:|---:|
|HR|61,189|7.74%|hour|729,026|84|73|96|23|
|O2Sat|95,079|12.03%|hour|695,136|98|96|99.5|3.5|
|Temp|523,314|66.22%|hour|266,901|37.06|36.56|37.55|0.99|
|SBP|120,201|15.21%|hour|670,014|118.5|105|134|29|
|MAP|80,858|10.23%|hour|709,357|77|68|87.33|19.33|
|DBP|380,297|48.13%|hour|409,918|58.5|51|67|16|
|Resp|77,258|9.78%|hour|712,957|18|15|22|7|
|EtCO2|790,215|100.00%|hour|0|not reported|not reported|not reported|not reported|
|BaseExcess|707,834|89.57%|hour|82,381|0|-3|1|4|
|HCO3|726,598|91.95%|hour|63,617|24|22|27|5|
|FiO2|678,060|85.81%|hour|112,155|0.5|0.4|0.55|0.15|
|pH|699,600|88.53%|hour|90,615|7.39|7.34|7.43|0.09|
|PaCO2|720,927|91.23%|hour|69,288|40|36|45|9|
|SaO2|751,055|95.04%|hour|39,160|97|93|98|5|
|AST|778,395|98.50%|hour|11,820|57|28|180|152|
|BUN|725,739|91.84%|hour|64,476|18|12|29|17|
|Alkalinephos|778,683|98.54%|hour|11,532|78|56|119|63|
|Calcium|750,897|95.02%|hour|39,318|8.3|7.8|8.8|1|
|Chloride|724,438|91.68%|hour|65,777|106|102|109|7|
|Creatinine|737,728|93.36%|hour|52,487|0.9|0.7|1.4|0.7|
|Bilirubin_direct|789,033|99.85%|hour|1,182|1.4|0.5|3.7|3.2|
|Glucose|693,559|87.77%|hour|96,656|124|104|150|46|
|Lactate|763,072|96.57%|hour|27,143|1.8|1.2|2.8|1.6|
|Magnesium|728,734|92.22%|hour|61,481|2|1.8|2.2|0.4|
|Phosphate|750,319|94.95%|hour|39,896|3.4|2.7|4.2|1.5|
|Potassium|704,379|89.14%|hour|85,836|4.1|3.8|4.5|0.7|
|Bilirubin_total|780,522|98.77%|hour|9,693|0.9|0.5|2.2|1.7|
|TroponinI|789,250|99.88%|hour|965|4.3|1|12.9|11.9|
|Hct|697,157|88.22%|hour|93,058|30.2|27.4|33.5|6.1|
|Hgb|720,394|91.16%|hour|69,821|10.4|9.4|11.6|2.2|
|PTT|751,909|95.15%|hour|38,306|32.4|27.7|42.9|15.2|
|WBC|730,867|92.49%|hour|59,348|10.8|8|14.3|6.3|
|Fibrinogen|784,185|99.24%|hour|6,030|250|184|356.8|172.8|
|Platelets|738,716|93.48%|hour|51,499|181|127|247|120|
|Age|0|0.00%|patient|20,336|64.67|52.29|75.66|23.37|
|Gender|0|0.00%|patient|20,336|1|0|1|1|
|Unit1|386,165|48.87%|patient|10,814|0|0|1|1|
|Unit2|386,165|48.87%|patient|10,814|1|0|1|1|
|HospAdmTime|8|0.00%|patient|20,335|-2.77|-34.13|-0.02|34.11|
|ICULOS|0|0.00%|hour|790,215|21|11|35|24|

### Hospital B

Age median [Q1, Q3]: 62 [50, 72]; missing: 0.

| Age category | Records |
|---|---:|
|<18|22|
|18–39|2,325|
|40–64|8,749|
|65–79|6,582|
|80–89|1,930|
|>=90 coded 100|392|
|>=90 other coding|0|
|Missing|0|

| Recorded Gender | Records |
|---|---:|
|Code 0|9,268|
|Code 1|10,732|
|Missing|0|
|Other code|0|

### Hospital B: predictor availability and distributions

Missingness always uses hourly rows. Distribution counts/quantiles are hour-weighted except the five demographic/administrative fields marked patient. ICULOS is hourly.

| Predictor | Missing hours | Missing fraction | Distribution unit | Observations | Median | Q1 | Q3 | IQR |
|---|---:|---:|---|---:|---:|---:|---:|---:|
|HR|92,210|12.10%|hour|669,785|83|71.5|95|23.5|
|O2Sat|107,657|14.13%|hour|654,338|98|96|99.5|3.5|
|Temp|503,670|66.10%|hour|258,325|36.9|36.5|37.4|0.9|
|SBP|106,064|13.92%|hour|655,931|124|109|142|33|
|MAP|112,412|14.75%|hour|649,583|84|74.5|96|21.5|
|DBP|106,257|13.94%|hour|655,738|65|56|74|18|
|Resp|161,077|21.14%|hour|600,918|18|16|21|5|
|EtCO2|704,359|92.44%|hour|57,636|33|28|38|10|
|BaseExcess|760,231|99.77%|hour|1,764|-2.4|-4.9|-0.2|4.7|
|HCO3|760,584|99.81%|hour|1,411|23.4|21.45|25.1|3.65|
|FiO2|744,785|97.74%|hour|17,210|0.4|0.4|0.6|0.2|
|pH|745,037|97.77%|hour|16,958|7.38|7.32|7.43|0.11|
|PaCO2|744,982|97.77%|hour|17,013|39|34.7|44|9.3|
|SaO2|747,594|98.11%|hour|14,401|97.5|95.7|98.7|3|
|AST|748,632|98.25%|hour|13,363|32|20|72|52|
|BUN|719,903|94.48%|hour|42,092|17|11|28|17|
|Alkalinephos|748,586|98.24%|hour|13,409|70|52|101|49|
|Calcium|709,982|93.17%|hour|52,013|8.2|7.2|8.7|1.5|
|Chloride|757,306|99.38%|hour|4,689|107|104|110|6|
|Creatinine|719,866|94.47%|hour|42,129|0.99|0.75|1.51|0.76|
|Bilirubin_direct|760,187|99.76%|hour|1,808|0.2|0.1|0.6825|0.5825|
|Glucose|593,135|77.84%|hour|168,860|128|108|155|47|
|Lactate|747,692|98.12%|hour|14,303|1.97|1.33|3.43|2.1|
|Magnesium|725,525|95.21%|hour|36,470|2|1.8|2.2|0.4|
|Phosphate|739,590|97.06%|hour|22,405|3.3|2.6|4|1.4|
|Potassium|703,306|92.30%|hour|58,689|4|3.7|4.4|0.7|
|Bilirubin_total|748,547|98.24%|hour|13,448|0.8|0.6|1.4|0.8|
|TroponinI|748,179|98.19%|hour|13,816|0.22|0.03|3.112|3.083|
|Hct|717,620|94.18%|hour|44,375|30.5|26|35.7|9.7|
|Hgb|717,225|94.12%|hour|44,770|9.9|8.5|11.7|3.2|
|PTT|754,602|99.03%|hour|7,393|32.3|28.5|42|13.5|
|WBC|721,896|94.74%|hour|40,099|9.6|7|12.8|5.8|
|Fibrinogen|757,783|99.45%|hour|4,212|251|183|341|158|
|Platelets|721,285|94.66%|hour|40,710|180|126|240|114|
|Age|0|0.00%|patient|20,000|62|50|72|22|
|Gender|0|0.00%|patient|20,000|1|0|1|1|
|Unit1|225,795|29.63%|patient|13,905|0|0|1|1|
|Unit2|225,795|29.63%|patient|13,905|1|0|1|1|
|HospAdmTime|0|0.00%|patient|20,000|-8.57|-52.35|-3.38|48.97|
|ICULOS|0|0.00%|hour|761,995|20|10|34|24|

## Definitions and limits

- **records:** Released patient records, not independently linked unique people or admissions.
- **stay_length:** Number of recorded hourly rows; not reconstructed total hospital or ICU stay.
- **event:** Ever positive supplied SepsisLabel; not the separate eligible pre-onset event count.
- **missingness:** Fraction of all recorded hourly rows with NULL or NaN; infinite values are rejected.
- **quantiles:** Exact continuous 25th, 50th, and 75th percentiles; IQR is Q3 minus Q1.
- **weighting:** Hourly observed values except Age, Gender, Unit1, Unit2 and HospAdmTime, which use each record's first nonmissing value. ICULOS remains hourly because it changes over time.
- **age:** Source Age=100 represents the >=90 category, not exact age 100. Numeric summaries retain source coding and must be interpreted accordingly.
- **gender:** Source recorded codes 0/1, plus missing/other; not inferred gender identity.
- **small_samples:** Quantiles and stay extrema are suppressed when fewer than 5 relevant observations/records are available; aggregate denominators remain reported.
- **scope:** Source characteristics only. No model scores, performance, raw identifiers, or individual trajectories are exported. Cohort scope is determined by the accompanying run manifest.

No identifier or row-level table is written. DuckDB uses two threads and a 1 GB memory limit; predictor quantiles are processed in batches of four.
