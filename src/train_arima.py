import neptune

run = neptune.init_run(
    project="bealtaine16/co2-emission", api_token="github/NEPTUNE_API_TOKEN"
)
neptune_api_token = run["github/NEPTUNE_API_TOKEN"]
print(neptune_api_token)
