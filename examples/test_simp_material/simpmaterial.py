import torchfea

class SIMPMaterial(torchfea.materials.Materials_Base):
    def __init__(self, ):
        super(SIMPMaterial, self).__init__()
        self.r = r

    def forward(self, x):
        return self.Emin + (self.E0 - self.Emin) * x ** self.p / (x ** self.p + self.q * (1 - x) ** self.r)