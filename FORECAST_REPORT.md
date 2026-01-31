# PandemicLLM Bio-Threat Forecast Report
Model: Gemma 3 270M (Custom Ordinal SFT)
Validation Samples: 10 random samples from unseen data

## Forecast 1: Georgia
**Hospitalization History:** `[55.70, 54.70, 46.40, 42.50]`  
**Actual Next-Week Outcome:** SUBSTANTIAL DECREASING (0)  

### Model Reasoning
**Predicted Category:** SUBSTANTIAL DECREASING  
**Predicted Index (Soft):** 0.3568  
**Probability Distribution:**  
- SUBSTANTIAL DECREASING:  77.7% ███████████████
- MODERATE DECREASING:  13.5% ██
- STABLE:   6.0% █
- MODERATE INCREASING:   1.3% 
- SUBSTANTIAL INCREASING:   1.6% 

**Raw Prompt Seen by Model:**
```xml
<bos><start_of_turn>user
You are an assistant that forecasts the trend of hospitalization for the next week for a state based on the information below:

<information>
	<Static>
		<state_name>Georgia</state_name>
		<Population>10617423</Population>
		<under_20>0.26</under_20>
		<over_65>0.14</over_65>
		<White>0.6</White>
		<Black>0.33</Black>
		<medicaid_coverage>0.173</medicaid_coverage>
		<poverty_rate>0.134</poverty_rate>
	</Static>
	<hospitalization_rate_history_4_weeks_chronological>[55.70, 54.70, 46.40, 42.50]</hospitalization_rate_history_4_weeks_chronological>
</information>

Now, predict the trend of hospitalization for the next week. Output 0 for Substantial Decreasing, 1 for Moderate Decreasing, 2 for Stable, 3 for Moderate Increasing, or 4 for Substantial Increasing.
The answer index is:<end_of_turn>
<start_of_turn>model

```

---

## Forecast 2: Delaware
**Hospitalization History:** `[12.30, 11.30, 10.40, 11.00]`  
**Actual Next-Week Outcome:** SUBSTANTIAL INCREASING (4)  

### Model Reasoning
**Predicted Category:** STABLE  
**Predicted Index (Soft):** 2.0769  
**Probability Distribution:**  
- SUBSTANTIAL DECREASING:   3.0% 
- MODERATE DECREASING:  17.0% ███
- STABLE:  55.7% ███████████
- MODERATE INCREASING:  18.1% ███
- SUBSTANTIAL INCREASING:   6.3% █

**Raw Prompt Seen by Model:**
```xml
<bos><start_of_turn>user
You are an assistant that forecasts the trend of hospitalization for the next week for a state based on the information below:

<information>
	<Static>
		<state_name>Delaware</state_name>
		<Population>973764</Population>
		<under_20>0.24</under_20>
		<over_65>0.19</over_65>
		<White>0.69</White>
		<Black>0.23</Black>
		<medicaid_coverage>0.204</medicaid_coverage>
		<poverty_rate>0.109</poverty_rate>
	</Static>
	<hospitalization_rate_history_4_weeks_chronological>[12.30, 11.30, 10.40, 11.00]</hospitalization_rate_history_4_weeks_chronological>
</information>

Now, predict the trend of hospitalization for the next week. Output 0 for Substantial Decreasing, 1 for Moderate Decreasing, 2 for Stable, 3 for Moderate Increasing, or 4 for Substantial Increasing.
The answer index is:<end_of_turn>
<start_of_turn>model

```

---

## Forecast 3: Alabama
**Hospitalization History:** `[9.00, 9.20, 8.60, 10.00]`  
**Actual Next-Week Outcome:** MODERATE DECREASING (1)  

### Model Reasoning
**Predicted Category:** MODERATE INCREASING  
**Predicted Index (Soft):** 2.7960  
**Probability Distribution:**  
- SUBSTANTIAL DECREASING:   0.2% 
- MODERATE DECREASING:   1.9% 
- STABLE:  27.7% █████
- MODERATE INCREASING:  58.7% ███████████
- SUBSTANTIAL INCREASING:  11.6% ██

**Raw Prompt Seen by Model:**
```xml
<bos><start_of_turn>user
You are an assistant that forecasts the trend of hospitalization for the next week for a state based on the information below:

<information>
	<Static>
		<state_name>Alabama</state_name>
		<Population>4903185</Population>
		<under_20>0.25</under_20>
		<over_65>0.17</over_65>
		<White>0.69</White>
		<Black>0.27</Black>
		<medicaid_coverage>0.195</medicaid_coverage>
		<poverty_rate>0.154</poverty_rate>
	</Static>
	<hospitalization_rate_history_4_weeks_chronological>[9.00, 9.20, 8.60, 10.00]</hospitalization_rate_history_4_weeks_chronological>
</information>

Now, predict the trend of hospitalization for the next week. Output 0 for Substantial Decreasing, 1 for Moderate Decreasing, 2 for Stable, 3 for Moderate Increasing, or 4 for Substantial Increasing.
The answer index is:<end_of_turn>
<start_of_turn>model

```

---

## Forecast 4: Iowa
**Hospitalization History:** `[6.40, 5.60, 4.80, 4.10]`  
**Actual Next-Week Outcome:** MODERATE DECREASING (1)  

### Model Reasoning
**Predicted Category:** STABLE  
**Predicted Index (Soft):** 1.7857  
**Probability Distribution:**  
- SUBSTANTIAL DECREASING:   2.0% 
- MODERATE DECREASING:  27.5% █████
- STABLE:  62.0% ████████████
- MODERATE INCREASING:   7.0% █
- SUBSTANTIAL INCREASING:   1.6% 

**Raw Prompt Seen by Model:**
```xml
<bos><start_of_turn>user
You are an assistant that forecasts the trend of hospitalization for the next week for a state based on the information below:

<information>
	<Static>
		<state_name>Iowa</state_name>
		<Population>3155070</Population>
		<under_20>0.26</under_20>
		<over_65>0.18</over_65>
		<White>0.91</White>
		<Black>0.04</Black>
		<medicaid_coverage>0.195</medicaid_coverage>
		<poverty_rate>0.114</poverty_rate>
	</Static>
	<hospitalization_rate_history_4_weeks_chronological>[6.40, 5.60, 4.80, 4.10]</hospitalization_rate_history_4_weeks_chronological>
</information>

Now, predict the trend of hospitalization for the next week. Output 0 for Substantial Decreasing, 1 for Moderate Decreasing, 2 for Stable, 3 for Moderate Increasing, or 4 for Substantial Increasing.
The answer index is:<end_of_turn>
<start_of_turn>model

```

---

## Forecast 5: Alabama
**Hospitalization History:** `[18.10, 16.90, 16.80, 16.40]`  
**Actual Next-Week Outcome:** MODERATE DECREASING (1)  

### Model Reasoning
**Predicted Category:** STABLE  
**Predicted Index (Soft):** 2.2863  
**Probability Distribution:**  
- SUBSTANTIAL DECREASING:   4.5% 
- MODERATE DECREASING:  16.6% ███
- STABLE:  42.4% ████████
- MODERATE INCREASING:  18.8% ███
- SUBSTANTIAL INCREASING:  17.7% ███

**Raw Prompt Seen by Model:**
```xml
<bos><start_of_turn>user
You are an assistant that forecasts the trend of hospitalization for the next week for a state based on the information below:

<information>
	<Static>
		<state_name>Alabama</state_name>
		<Population>4903185</Population>
		<under_20>0.25</under_20>
		<over_65>0.17</over_65>
		<White>0.69</White>
		<Black>0.27</Black>
		<medicaid_coverage>0.195</medicaid_coverage>
		<poverty_rate>0.154</poverty_rate>
	</Static>
	<hospitalization_rate_history_4_weeks_chronological>[18.10, 16.90, 16.80, 16.40]</hospitalization_rate_history_4_weeks_chronological>
</information>

Now, predict the trend of hospitalization for the next week. Output 0 for Substantial Decreasing, 1 for Moderate Decreasing, 2 for Stable, 3 for Moderate Increasing, or 4 for Substantial Increasing.
The answer index is:<end_of_turn>
<start_of_turn>model

```

---

## Forecast 6: North Dakota
**Hospitalization History:** `[6.00, 8.80, 13.30, 9.40]`  
**Actual Next-Week Outcome:** STABLE (2)  

### Model Reasoning
**Predicted Category:** MODERATE INCREASING  
**Predicted Index (Soft):** 2.9847  
**Probability Distribution:**  
- SUBSTANTIAL DECREASING:   0.1% 
- MODERATE DECREASING:   1.7% 
- STABLE:  17.5% ███
- MODERATE INCREASING:  60.9% ████████████
- SUBSTANTIAL INCREASING:  19.8% ███

**Raw Prompt Seen by Model:**
```xml
<bos><start_of_turn>user
You are an assistant that forecasts the trend of hospitalization for the next week for a state based on the information below:

<information>
	<Static>
		<state_name>North Dakota</state_name>
		<Population>762062</Population>
		<under_20>0.26</under_20>
		<over_65>0.16</over_65>
		<White>0.87</White>
		<Black>0.03</Black>
		<medicaid_coverage>0.122</medicaid_coverage>
		<poverty_rate>0.11</poverty_rate>
	</Static>
	<hospitalization_rate_history_4_weeks_chronological>[6.00, 8.80, 13.30, 9.40]</hospitalization_rate_history_4_weeks_chronological>
</information>

Now, predict the trend of hospitalization for the next week. Output 0 for Substantial Decreasing, 1 for Moderate Decreasing, 2 for Stable, 3 for Moderate Increasing, or 4 for Substantial Increasing.
The answer index is:<end_of_turn>
<start_of_turn>model

```

---

## Forecast 7: New Hampshire
**Hospitalization History:** `[6.50, 9.00, 6.90, 8.50]`  
**Actual Next-Week Outcome:** MODERATE INCREASING (3)  

### Model Reasoning
**Predicted Category:** STABLE  
**Predicted Index (Soft):** 2.1306  
**Probability Distribution:**  
- SUBSTANTIAL DECREASING:   1.0% 
- MODERATE DECREASING:  14.2% ██
- STABLE:  59.8% ███████████
- MODERATE INCREASING:  20.7% ████
- SUBSTANTIAL INCREASING:   4.3% 

**Raw Prompt Seen by Model:**
```xml
<bos><start_of_turn>user
You are an assistant that forecasts the trend of hospitalization for the next week for a state based on the information below:

<information>
	<Static>
		<state_name>New Hampshire</state_name>
		<Population>1359711</Population>
		<under_20>0.21</under_20>
		<over_65>0.19</over_65>
		<White>0.93</White>
		<Black>0.02</Black>
		<medicaid_coverage>0.132</medicaid_coverage>
		<poverty_rate>0.075</poverty_rate>
	</Static>
	<hospitalization_rate_history_4_weeks_chronological>[6.50, 9.00, 6.90, 8.50]</hospitalization_rate_history_4_weeks_chronological>
</information>

Now, predict the trend of hospitalization for the next week. Output 0 for Substantial Decreasing, 1 for Moderate Decreasing, 2 for Stable, 3 for Moderate Increasing, or 4 for Substantial Increasing.
The answer index is:<end_of_turn>
<start_of_turn>model

```

---

## Forecast 8: Louisiana
**Hospitalization History:** `[64.70, 60.70, 53.90, 41.90]`  
**Actual Next-Week Outcome:** SUBSTANTIAL DECREASING (0)  

### Model Reasoning
**Predicted Category:** SUBSTANTIAL DECREASING  
**Predicted Index (Soft):** 0.0354  
**Probability Distribution:**  
- SUBSTANTIAL DECREASING:  97.6% ███████████████████
- MODERATE DECREASING:   1.7% 
- STABLE:   0.5% 
- MODERATE INCREASING:   0.1% 
- SUBSTANTIAL INCREASING:   0.1% 

**Raw Prompt Seen by Model:**
```xml
<bos><start_of_turn>user
You are an assistant that forecasts the trend of hospitalization for the next week for a state based on the information below:

<information>
	<Static>
		<state_name>Louisiana</state_name>
		<Population>4648794</Population>
		<under_20>0.26</under_20>
		<over_65>0.16</over_65>
		<White>0.63</White>
		<Black>0.33</Black>
		<medicaid_coverage>0.293</medicaid_coverage>
		<poverty_rate>0.187</poverty_rate>
	</Static>
	<hospitalization_rate_history_4_weeks_chronological>[64.70, 60.70, 53.90, 41.90]</hospitalization_rate_history_4_weeks_chronological>
</information>

Now, predict the trend of hospitalization for the next week. Output 0 for Substantial Decreasing, 1 for Moderate Decreasing, 2 for Stable, 3 for Moderate Increasing, or 4 for Substantial Increasing.
The answer index is:<end_of_turn>
<start_of_turn>model

```

---

## Forecast 9: New Hampshire
**Hospitalization History:** `[9.50, 9.30, 9.00, 11.80]`  
**Actual Next-Week Outcome:** MODERATE INCREASING (3)  

### Model Reasoning
**Predicted Category:** MODERATE INCREASING  
**Predicted Index (Soft):** 2.9485  
**Probability Distribution:**  
- SUBSTANTIAL DECREASING:   0.1% 
- MODERATE DECREASING:   1.8% 
- STABLE:  21.8% ████
- MODERATE INCREASING:  55.8% ███████████
- SUBSTANTIAL INCREASING:  20.5% ████

**Raw Prompt Seen by Model:**
```xml
<bos><start_of_turn>user
You are an assistant that forecasts the trend of hospitalization for the next week for a state based on the information below:

<information>
	<Static>
		<state_name>New Hampshire</state_name>
		<Population>1359711</Population>
		<under_20>0.21</under_20>
		<over_65>0.19</over_65>
		<White>0.93</White>
		<Black>0.02</Black>
		<medicaid_coverage>0.132</medicaid_coverage>
		<poverty_rate>0.075</poverty_rate>
	</Static>
	<hospitalization_rate_history_4_weeks_chronological>[9.50, 9.30, 9.00, 11.80]</hospitalization_rate_history_4_weeks_chronological>
</information>

Now, predict the trend of hospitalization for the next week. Output 0 for Substantial Decreasing, 1 for Moderate Decreasing, 2 for Stable, 3 for Moderate Increasing, or 4 for Substantial Increasing.
The answer index is:<end_of_turn>
<start_of_turn>model

```

---

## Forecast 10: Delaware
**Hospitalization History:** `[17.70, 15.40, 13.90, 9.00]`  
**Actual Next-Week Outcome:** SUBSTANTIAL DECREASING (0)  

### Model Reasoning
**Predicted Category:** SUBSTANTIAL DECREASING  
**Predicted Index (Soft):** 0.2702  
**Probability Distribution:**  
- SUBSTANTIAL DECREASING:  81.4% ████████████████
- MODERATE DECREASING:  12.5% ██
- STABLE:   4.6% 
- MODERATE INCREASING:   0.9% 
- SUBSTANTIAL INCREASING:   0.7% 

**Raw Prompt Seen by Model:**
```xml
<bos><start_of_turn>user
You are an assistant that forecasts the trend of hospitalization for the next week for a state based on the information below:

<information>
	<Static>
		<state_name>Delaware</state_name>
		<Population>973764</Population>
		<under_20>0.24</under_20>
		<over_65>0.19</over_65>
		<White>0.69</White>
		<Black>0.23</Black>
		<medicaid_coverage>0.204</medicaid_coverage>
		<poverty_rate>0.109</poverty_rate>
	</Static>
	<hospitalization_rate_history_4_weeks_chronological>[17.70, 15.40, 13.90, 9.00]</hospitalization_rate_history_4_weeks_chronological>
</information>

Now, predict the trend of hospitalization for the next week. Output 0 for Substantial Decreasing, 1 for Moderate Decreasing, 2 for Stable, 3 for Moderate Increasing, or 4 for Substantial Increasing.
The answer index is:<end_of_turn>
<start_of_turn>model

```

---

