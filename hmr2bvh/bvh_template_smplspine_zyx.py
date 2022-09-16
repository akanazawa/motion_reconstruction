"""
Spine is more complex but it's what SMPL has.
"""
template = """HIERARCHY
ROOT Hips
{{
	OFFSET {d[0]} {d[1]} {d[2]}
	CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
	JOINT LeftHip
	{{
		OFFSET {d[3]} {d[4]} {d[5]}
		CHANNELS 3 Zrotation Yrotation Xrotation
		JOINT LeftKnee
		{{
			OFFSET {d[6]} {d[7]} {d[8]}
			CHANNELS 3 Zrotation Yrotation Xrotation
			JOINT LeftFoot
			{{
				OFFSET {d[9]} {d[10]} {d[11]}
				CHANNELS 3 Zrotation Yrotation Xrotation
				End LeftToe
				{{
                                  OFFSET {d[12]} {d[13]} {d[14]}
				}}
			}}
		}}
	}}
	JOINT RightHip
	{{
		OFFSET {d[15]} {d[16]} {d[17]}
		CHANNELS 3 Zrotation Yrotation Xrotation
		JOINT RightKnee
		{{
			OFFSET {d[18]} {d[19]} {d[20]}
			CHANNELS 3 Zrotation Yrotation Xrotation
			JOINT RightFoot
			{{
				OFFSET {d[21]} {d[22]} {d[23]}
				CHANNELS 3 Zrotation Yrotation Xrotation
				End RightToe
				{{
                                        OFFSET {d[24]} {d[25]} {d[26]}
				}}
			}}
		}}
	}}
	JOINT Waist
	{{
		OFFSET {d[27]} {d[28]} {d[29]}
		CHANNELS 3 Zrotation Yrotation Xrotation
		JOINT Spine
		{{
			OFFSET {d[30]} {d[31]} {d[32]}
			CHANNELS 3 Zrotation Yrotation Xrotation
                        JOINT Chest
                        {{
                            OFFSET {d[33]} {d[34]} {d[35]}
                            CHANNELS 3 Zrotation Yrotation Xrotation
                            JOINT Neck
                            {{
                                OFFSET {d[36]} {d[37]} {d[38]}
                                CHANNELS 3 Zrotation Yrotation Xrotation
                                End Head
                                {{
                                        OFFSET {d[39]} {d[40]} {d[41]}
                                }}
                            }}
                            JOINT LeftInnerShoulder
                            {{
                                OFFSET {d[42]} {d[43]} {d[44]}
                                CHANNELS 3 Zrotation Yrotation Xrotation
                                JOINT LeftShoulder
                                {{
                                        OFFSET {d[45]} {d[46]} {d[47]}
                                        CHANNELS 3 Zrotation Yrotation Xrotation
                                        JOINT LeftElbow
                                        {{
                                                OFFSET {d[48]} {d[49]} {d[50]}
                                                CHANNELS 3 Zrotation Yrotation Xrotation
                                                End LeftWrist
                                                {{
                                                        OFFSET {d[51]} {d[52]} {d[53]}
                                                }}
                                        }}
                                }}
                            }}
                            JOINT RightInnerShoulder
                            {{
                                OFFSET {d[54]} {d[55]} {d[56]}
                                CHANNELS 3 Zrotation Yrotation Xrotation
                                JOINT RightShoulder
                                {{
                                        OFFSET {d[57]} {d[58]} {d[59]}
                                        CHANNELS 3 Zrotation Yrotation Xrotation
                                        JOINT RightElbow
                                        {{
                                                OFFSET {d[60]} {d[61]} {d[62]}
                                                CHANNELS 3 Zrotation Yrotation Xrotation
                                                End RightWrist
                                                {{
                                                        OFFSET {d[63]} {d[64]} {d[65]}
                                                }}
                                        }}
                                }}
                            }}
                        }}
                }}
            }}
}}
MOTION
Frames: {d[66]}
Frame Time: {d[67]}
"""
