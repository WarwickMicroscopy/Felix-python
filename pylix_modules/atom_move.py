import numpy as np

def atom_move(space_group_number, wyckoff):
    moves = np.zeros([3, 3])
    if space_group_number == 1:  # P1
        if ('x') in wyckoff:
            moves[0, :] = ([0.0, 1.0, 0.0])
            moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('b') in wyckoff:
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        else:
            raise ValueError("Wyckoff Symbol for space group P1 not recognised")   
    # if space_group_number == 2)
    # if space_group_number == 3)
    # if space_group_number == 4)
    # if space_group_number == 5)
    # if space_group_number == 6)
    # if space_group_number == 7)
    # if space_group_number == 8)
    # if space_group_number == 9)
    # if space_group_number == 10)
    # if space_group_number == 11)
    # if space_group_number == 12)
    # if space_group_number == 13)
    # if space_group_number == 14)
    # if space_group_number == 15)
    if space_group_number == 15:  # C 1 2/c 1
        if ('e') in wyckoff:  # point symmetry 2 along y
            moves[0, :] = ([0.0, 1.0, 0.0])
        elif ('f') in wyckoff:  # point symmetry 1
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'c' or 'd') not in wyckoff:  # point symmetry -1
            raise ValueError("Wyckoff Symbol for space group C2/c, not recognised")
    # if space_group_number == 16)
    # if space_group_number == 17)
    # if space_group_number == 18)
    # if space_group_number == 19)
    # if space_group_number == 20)
    # if space_group_number == 21)
    # if space_group_number == 22)
    # if space_group_number == 23)
    # if space_group_number == 24)
    # if space_group_number == 25)
    # if space_group_number == 26)
    # if space_group_number == 27)
    # if space_group_number == 28)
    # if space_group_number == 29)
    # if space_group_number == 30)
    # if space_group_number == 31)
    # if space_group_number == 32)
    # if space_group_number == 33)
    # if space_group_number == 34)
    # if space_group_number == 35)
    if space_group_number == 36:  # C m c 21
        # NEED TO CODE ALTERNATIVE SETTINGS Ccm21,Bb21m,Bm21b,A21ma,A21am
        if ('a') in wyckoff:  # point symmetry 1, coordinate [x,y,z],
          moves[0, :] = ([0.0, 1.0, 0.0])
          moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('b') in wyckoff:  # point symmetry 1, coordinate [x,y,z],
          moves[0, :] = ([1.0, 0.0, 0.0])
          moves[1, :] = ([0.0, 1.0, 0.0])
          moves[2, :] = ([0.0, 0.0, 1.0])
        else:
            raise ValueError("Wyckoff Symbol for space group Cmc21 not recognised")
    # if space_group_number == 37)
    # if space_group_number == 38)
    # if space_group_number == 39)
    # if space_group_number == 40)
    # if space_group_number == 41)
    # if space_group_number == 42)
    # if space_group_number == 43)
    # if space_group_number == 44)
    # if space_group_number == 45)
    # if space_group_number == 46)
    # if space_group_number == 47)
    # if space_group_number == 48)
    # if space_group_number == 49)
    # if space_group_number == 50)
    # if space_group_number == 51)
    # if space_group_number == 52)
    # if space_group_number == 53)
    # if space_group_number == 54)
    # if space_group_number == 55)
    # if space_group_number == 56)
    # if space_group_number == 57)
    # if space_group_number == 58)
    # if space_group_number == 59)
    # if space_group_number == 60)
    # if space_group_number == 61)
    # if space_group_number == 62)
    if space_group_number == 63:  # Cmcm
        # a: point symmetry 2/m, coordinate [0,0,0] & eq, no movement
        # b: point symmetry 2/m, coordinate [0,1/2,0] & eq, no movement
        # d: point symmetry -1, coordinate [1/4,1/4,0] & eq, no movement
        if ('c') in wyckoff:  # point symmetry mm2, coordinate [0,y,1/4] & eq
            moves[0, :] = ([0.0, 1.0, 0.0])
        elif ('e') in wyckoff:  # point symmetry 2(x), coordinate [x,0,0] & eq
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('f') in wyckoff:  # point symmetry m(x), coordinate [0,y,z] & eq
            moves[0, :] = ([0.0, 1.0, 0.0])
            moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('g') in wyckoff:  # point symmetry m(z), coordinate [x,y,1/4] & eq
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
        elif ('h') in wyckoff:  # point symmetry 1, coordinate [x,y,z] & eq
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'd') not in wyckoff:
            raise ValueError("Wyckoff Symbol for space group Cmc2m not recognised")
    if space_group_number == 64:  # Cmca
        # a: point symmetry 2/m, coordinate [0,0,0] & eq, no movement
        # b: point symmetry 2/m, coordinate [1/2,0,0] & eq, no movement
        # c: point symmetry -1, coordinate [1/4,1/4,0] & eq, no movement
        if ('d') in wyckoff:  # point symmetry 2(x), coordinate [x,0,0] & eq
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('e') in wyckoff:  # point symmetry 2(y), coordinate [1/4,y,1/4] & eq
            moves[0, :] = ([0.0, 1.0, 0.0])
        elif ('f') in wyckoff:  # point symmetry m(x), coordinate [0,y,z] & eq
            moves[0, :] = ([0.0, 1.0, 0.0])
            moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('g') in wyckoff:  # point symmetry 1, coordinate [x,y,z] & eq
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'c') not in wyckoff:
            raise ValueError("Wyckoff Symbol for space group Cmca not recognised")
    # if space_group_number == 65)
    # if space_group_number == 66)
    # if space_group_number == 67)
    if space_group_number == 68:  # Ccca
        # N.B. multiple origin choices allowed, here origin at 222, -1 at [1/4,0,1/4]
        # a: point symmetry 222, coordinate [0,0,0] & eq, no movement
        # b: point symmetry 222, coordinate [0,0,1/2] & eq, no movement
        # c: point symmetry -1, coordinate [1/4,0,1/4] & eq, no movement
        # d: point symmetry -1, coordinate [0,1/4,1/4] & eq, no movement
        if ('e') in wyckoff:  # point symmetry 2(x), coordinate [x,0,0] & eq
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('f') in wyckoff:  # point symmetry 2(y), coordinate [0,y,0] & eq
            moves[0, :] = ([0.0, 1.0, 0.0])
        elif ('g') in wyckoff:  # point symmetry 2(z), coordinate [0,0,z] & eq
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('h') in wyckoff:  # point symmetry 2(z), coordinate [1/4,1/4,z] & eq
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('i') in wyckoff:  # point symmetry 1, coordinate [x,y,z] & eq
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'c' or 'd') not in wyckoff:
            raise ValueError("Wyckoff Symbol for space group Ccca not recognised")
    # if space_group_number == 69)
    # if space_group_number == 70)
    # if space_group_number == 71)
    # if space_group_number == 72)
    # if space_group_number == 73)
    # if space_group_number == 74)
    # if space_group_number == 75)
    # if space_group_number == 76)
    # if space_group_number == 77)
    # if space_group_number == 78)
    # if space_group_number == 79)
    # if space_group_number == 80)
    # if space_group_number == 81)
    # if space_group_number == 82)
    # if space_group_number == 83)
    # if space_group_number == 84)
    # if space_group_number == 85)
    # if space_group_number == 86)
    # if space_group_number == 87)
    # if space_group_number == 88)
    # if space_group_number == 89)
    # if space_group_number == 90)
    # if space_group_number == 91)
    # if space_group_number == 92)
    # if space_group_number == 93)
    # if space_group_number == 94)
    # if space_group_number == 95)
    # if space_group_number == 96)
    # if space_group_number == 97)
    # if space_group_number == 98)
    if space_group_number == 99:  # P 4 m m 
        if ('a') in wyckoff:  # point symmetry 4mm, coordinate [0,0,z], allowed movement along [001]
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('b') in wyckoff:  # point symmetry 4mm, coordinate [1/2,1/2,z], allowed movement along [001]
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('c') in wyckoff:  # point symmetry mm, coordinate [1/2,0,z] & eq, allowed movement along [001]
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('d') in wyckoff:  # point symmetry m, coordinate [x,x,z] & eq, allowed movement along [110] & [001]
            moves[0, :] = ([1/np.sqrt(2), 1/np.sqrt(2), 0.0])
            moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('e') in wyckoff:  # point symmetry m, coordinate [x,0,z] & eq, allowed movement along [100] & [001]
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('f') in wyckoff:  # point symmetry m, coordinate [x,1/2,z] & eq, allowed movement along [100] & [001]
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('g') in wyckoff:  # point symmetry 1, coordinate [x,y,z] & eq
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        else:
            raise ValueError("Wyckoff Symbol for space group P4mm not recognised")
    # if space_group_number == 100)
    # if space_group_number == 101)
    # if space_group_number == 102)
    # if space_group_number == 103)
    # if space_group_number == 104)
    # if space_group_number == 105)
    # if space_group_number == 106)
    # if space_group_number == 107)
    # if space_group_number == 108)
    # if space_group_number == 109)
    # if space_group_number == 110)
    # if space_group_number == 111)
    # if space_group_number == 112)
    # if space_group_number == 113)
    # if space_group_number == 114)
    # if space_group_number == 115)
    # if space_group_number == 116)
    # if space_group_number == 117)
    # if space_group_number == 118)
    # if space_group_number == 119)
    # if space_group_number == 120)
    # if space_group_number == 121)
    # if space_group_number == 122)
    # if space_group_number == 123)
    # if space_group_number == 124)
    # if space_group_number == 125)
    # if space_group_number == 126)
    # if space_group_number == 127)
    # if space_group_number == 128)
    # if space_group_number == 129)
    # if space_group_number == 130)
    # if space_group_number == 131)
    # if space_group_number == 132)
    # if space_group_number == 133)
    # if space_group_number == 134)
    # if space_group_number == 135)
    # if space_group_number == 136)
    # if space_group_number == 137)
    # if space_group_number == 138)
    if space_group_number == 139:  # I4/m m m
        # a: point symmetry 4/mmm, coordinate [0,0,0], no allowed movements
        # b: point symmetry 4/mmm, coordinate [0,0,1/2], no allowed movements
        # c: point symmetry mmm, coordinate [0,1/2,0] or [1/2,0,0], no allowed movements
        # d: point symmetry -4m2, coordinate [0,1/2,1/4] or [1/2,0,1/4], no allowed movements
        # f: point symmetry 2/m, coordinate [1/4,1/4,1/4] & eq, no allowed movements
        if ('e') in wyckoff:  # point symmetry 4mm, coordinate [0,0,z] allowed movement along z
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('g') in wyckoff:  # point symmetry mm, coordinate [0,1/2,z] & eq, allowed movement along z
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('h') in wyckoff:  # point symmetry mm, coordinate [x,x,0] & eq, allowed movement along [110]
            moves[0, :] = ([1/np.sqrt(2), 1/np.sqrt(2), 0.0])
        elif ('i') in wyckoff:  # point symmetry mm, coordinate [x,0,0] & eq, allowed movement along [100]
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('j') in wyckoff:  # point symmetry mm, coordinate [x,1/2,0] & eq, allowed movement along [100]
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('k') in wyckoff:  # point symmetry 2, coordinate [x,1/2+x,1/4] & eq, allowed movement along [110]
            moves[0, :] = ([1/np.sqrt(2), 1/np.sqrt(2), 0.0])
        elif ('l') in wyckoff:  # point symmetry m, coordinate [x,y,0] & eq, allowed movement along [100] & [010]
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
        elif ('m') in wyckoff:  # point symmetry m, coordinate [x,x,z] & eq, allowed movement along [110] & [001]
            moves[0, :] = ([1/np.sqrt(2), 1/np.sqrt(2), 0.0])
            moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('n') in wyckoff:  # point symmetry m, coordinate [x,0,z] & eq, allowed movement along [100] & [001]
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('o') in wyckoff:  # point symmetry 1, coordinate [x,y,z] & eq, allowed movement along [100], [010] & [001]
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'c' or 'd' or 'f') not in wyckoff:
            raise ValueError("Wyckoff Symbol for space group I4/mmm not recognised")
    # if space_group_number == 140)
    # if space_group_number == 141)
    if space_group_number == 142:  # I41/acd
        # a: point symmetry -4, no allowed movements
        # b: point symmetry 222, no allowed movements
        # c: point symmetry -1, no allowed movements
        if ('d') in wyckoff:  # point symmetry 2, allowed movement along z
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('e') in wyckoff:  # point symmetry 2, allowed movement along x
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('f') in wyckoff:  # point symmetry 2, allowed movement along [x,x,0]
            moves[0, :] = ([1/np.sqrt(2), 1/np.sqrt(2), 0.0])
        elif ('g') in wyckoff:  # point symmetry 1
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'c') not in wyckoff:
            raise ValueError("Wyckoff Symbol for space group I41/acd not recognised")
    # if space_group_number == 143)
    # if space_group_number == 144)
    # if space_group_number == 145)
    # if space_group_number == 146)
    # if space_group_number == 147)
    # if space_group_number == 148)
    # if space_group_number == 149)
    # if space_group_number == 150)
    # if space_group_number == 151)
    # if space_group_number == 152)
    # if space_group_number == 153)
    # if space_group_number == 154)
    # if space_group_number == 155)
    # if space_group_number == 156)
    # if space_group_number == 157)
    # if space_group_number == 158)
    # if space_group_number == 159)
    # if space_group_number == 160)
    # if space_group_number == 161)
    # if space_group_number == 162)
    # if space_group_number == 163)
    # if space_group_number == 164)
    # if space_group_number == 165)
    # if space_group_number == 166)
    # if space_group_number == 167)
    if space_group_number == 167:  # R-3c
        # a: point symmetry 32, no allowed movements
        # b: point symmetry -3, no allowed movements
        # d: point symmetry -1, no allowed movements
        if ('c') in wyckoff:  # point symmetry 3, allowed movement along z
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('e') in wyckoff:  # point symmetry 2, allowed movement along [x,0,0]
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('f') in wyckoff:  # point symmetry 1
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'd') not in wyckoff:
            raise ValueError("Wyckoff Symbol for space group R-3c not recognised")
    # if space_group_number == 168)
    # if space_group_number == 169)
    # if space_group_number == 170)
    # if space_group_number == 171)
    # if space_group_number == 172)
    # if space_group_number == 173)
    # if space_group_number == 174)
    # if space_group_number == 175)
    # if space_group_number == 176)
    # if space_group_number == 177)
    # if space_group_number == 178)
    # if space_group_number == 179)
    # if space_group_number == 180)
    # if space_group_number == 181)
    # if space_group_number == 182)
    # if space_group_number == 183)
    # if space_group_number == 184)
    # if space_group_number == 185)
    # if space_group_number == 186)
    # if space_group_number == 187)
    # if space_group_number == 188)
    # if space_group_number == 189)
    # if space_group_number == 190)
    # if space_group_number == 191)
    # if space_group_number == 192)
    # if space_group_number == 193)
    # if space_group_number == 194)
    # if space_group_number == 195)
    # if space_group_number == 196)
    # if space_group_number == 197)
    # if space_group_number == 198)
    # if space_group_number == 199)
    # if space_group_number == 200)
    # if space_group_number == 201)
    # if space_group_number == 202)
    # if space_group_number == 203)
    # if space_group_number == 204)
    # if space_group_number == 205)
    # if space_group_number == 206)
    # if space_group_number == 207)
    # if space_group_number == 208)
    # if space_group_number == 209)
    # if space_group_number == 210)
    # if space_group_number == 211)
    # if space_group_number == 212)
    # if space_group_number == 213)
    # if space_group_number == 214)
    # if space_group_number == 215)
    if space_group_number == 216:  # F-43m
        # a: point symmetry -43m, coordinate [0,0,0], no allowed movements
        # b: point symmetry -43m, coordinate [1/2,1/2,1/2], no allowed movements
        # c: point symmetry -43m, coordinate [1/4,1/4,1/4], no allowed movements
        # d: point symmetry -43m, coordinate [3/4,3/4,3/4], no allowed movements
        if ('e') in wyckoff:  # point symmetry 3m, coordinate [x,x,x] allowed movement along [111]
            moves[0, :] = ([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
        elif ('f') in wyckoff:  # point symmetry mm, coordinate [x,0,0] allowed movements along x
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('g') in wyckoff:  # point symmetry mm, coordinate [x,1/4,1/4] allowed movement along x
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('h') in wyckoff:  # point symmetry m, coordinate [x,x,z], allowed movement along [110] and [001]
            moves[0, :] = ([1/np.sqrt(2), 1/np.sqrt(2), 0.0])
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('i') in wyckoff:  # point symmetry 1, coordinate [x,y,z], allowed movement along x,y,z
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'c' or 'd') not in wyckoff:
            raise ValueError("Wyckoff Symbol for space group F-43m not recognised")
    # if space_group_number == 217)
    # if space_group_number == 218)
    # if space_group_number == 219)
    # if space_group_number == 220)
    # if space_group_number == 221)
    # if space_group_number == 222)
    # if space_group_number == 223)
    # if space_group_number == 224)
    # if space_group_number == 225)
    # if space_group_number == 226)
    # if space_group_number == 227)
    # if space_group_number == 228)
    # if space_group_number == 229)
    # if space_group_number == 230)
    
    return moves
