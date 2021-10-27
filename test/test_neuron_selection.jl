using ExplainabilityMethods: MaxActivationNS, IndexNS

A = [-2.1694243, 2.4023275, 0.99464744, -0.1514646, 1.0307171]
ns1 = MaxActivationNS()
ns2 = IndexNS(4)

ns1(A) == 2
ns2(A) == 4
