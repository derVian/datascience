# datascience practices
--Stationarity Fix: Applied a Log Transformation to the passenger counts to stabilize increasing variance over time.
--Temporal Features: Extracted Year, Month, and One-Hot Encoded Seasons (Winter, Spring, Summer, Autumn) to capture annual patterns.
--Historical Memory: Created Lag Features (lag_1 and lag_12) to provide the model with short-term momentum and long-term seasonal context.
--Trend Capture: Integrated a trend (time-step) column to help the model map the linear growth of the airline industry.
