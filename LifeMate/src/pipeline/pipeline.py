from src.stages.stage1_perception import Stage1
from src.stages.stage2_planning import Stage2
from src.stages.stage3_execution import Stage3

class POMDPPipeline:
    def __init__(self, config):
        self.s1 = Stage1(config["stage1"])
        self.s2 = Stage2(config["stage2"])
        self.s3 = Stage3(config["stage3"])
        
    def run_sample(self, sample, env, catalog):
        s1_out = self.s1.run(sample, catalog)
        return self.s3.execute(env, self.s2, s1_out, sample["context"])