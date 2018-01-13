from a3c_agent_discrete import A3C_Workers

env_name = 'Pong-v0'
worker = A3C_Workers(env_name, test_mode=True)
worker.test_the_network('saved_models/Pong-v0/policy_model_80.pt')


