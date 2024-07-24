export COMET_API_KEY=""  # Your actual Comet API key

# Run the benchmark script for each profile
for p in v1-empty v1-scalars v1-tables
do
    python ./bench_mlflow.py --test_profile "$p"
done

