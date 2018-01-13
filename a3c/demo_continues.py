from a3c_agent_continues import A3C_Workers

env_name = 'Pendulum-v0'
worker = A3C_Workers(env_name)
worker.test_the_network('saved_models/Pendulum-v0/policy_model_6000.pt')