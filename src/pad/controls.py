"""Matched nonlinear additive controls on the same prepared inputs."""
import kernel as v4
RBF_W=[1.,0.,.5,1/6,5/6,1/3,2/3]
class Control(v4.Data):
    def __init__(self,name,family):
        super().__init__(name);self.family=family
        z=self.z
        self.P=v4.rbf(z['xp'],z['xp']);self.Pr=v4.rbf(z['xpr'],z['xp'])
        self.D=v4.rbf(z['xd'],z['xd']);self.Dr=v4.rbf(z['xdr'],z['xd'])
        self.weights=v4.WEIGHTS if family=='A_match' else [(w,1-w) for w in RBF_W]
    def components(self,tr):
        if self.family=='A_rbf':return [self.P,self.D],[self.Pr,self.Dr]
        c,r=super().components(tr)
        return [c[0],c[1],(self.P+self.D)/2],[r[0],r[1],(self.Pr+self.Dr)/2]
