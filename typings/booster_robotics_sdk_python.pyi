"""

        python binding of booster robotics sdk
        -----------------------
    
"""
from __future__ import annotations
import collections.abc
import typing
__all__ = ['B1HandAction', 'B1HandIndex', 'B1JointCnt', 'B1JointIndex', 'B1LocoApiId', 'B1LocoClient', 'B1LowCmdPublisher', 'B1LowHandDataScriber', 'B1LowStateSubscriber', 'B1OdometerStateSubscriber', 'ChannelFactory', 'DexterousFingerParameter', 'Frame', 'GetModeResponse', 'GripperControlMode', 'GripperMotionParameter', 'HandReplyData', 'HandReplyParam', 'ImuState', 'LowCmd', 'LowCmdType', 'LowState', 'MotorCmd', 'MotorState', 'Odometer', 'Orientation', 'PARALLEL', 'Position', 'Posture', 'Quaternion', 'RobotMode', 'SERIAL', 'Transform', 'kBody', 'kChangeMode', 'kCrankDownLeft', 'kCrankDownRight', 'kCrankUpLeft', 'kCrankUpRight', 'kCustom', 'kDamping', 'kForce', 'kHandClose', 'kHandOpen', 'kHead', 'kHeadPitch', 'kHeadYaw', 'kLeftElbowPitch', 'kLeftElbowYaw', 'kLeftFoot', 'kLeftHand', 'kLeftHipPitch', 'kLeftHipRoll', 'kLeftHipYaw', 'kLeftKneePitch', 'kLeftShoulderPitch', 'kLeftShoulderRoll', 'kMove', 'kPosition', 'kPrepare', 'kRightElbowPitch', 'kRightElbowYaw', 'kRightFoot', 'kRightHand', 'kRightHipPitch', 'kRightHipRoll', 'kRightHipYaw', 'kRightKneePitch', 'kRightShoulderPitch', 'kRightShoulderRoll', 'kRotateHead', 'kUnknown', 'kWaist', 'kWalking']
class B1HandAction:
    """
    Members:
    
      kHandOpen
    
      kHandClose
    """
    __members__: typing.ClassVar[dict[str, B1HandAction]]  # value = {'kHandOpen': <B1HandAction.kHandOpen: 0>, 'kHandClose': <B1HandAction.kHandClose: 1>}
    kHandClose: typing.ClassVar[B1HandAction]  # value = <B1HandAction.kHandClose: 1>
    kHandOpen: typing.ClassVar[B1HandAction]  # value = <B1HandAction.kHandOpen: 0>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class B1HandIndex:
    """
    Members:
    
      kLeftHand
    
      kRightHand
    """
    __members__: typing.ClassVar[dict[str, B1HandIndex]]  # value = {'kLeftHand': <B1HandIndex.kLeftHand: 0>, 'kRightHand': <B1HandIndex.kRightHand: 1>}
    kLeftHand: typing.ClassVar[B1HandIndex]  # value = <B1HandIndex.kLeftHand: 0>
    kRightHand: typing.ClassVar[B1HandIndex]  # value = <B1HandIndex.kRightHand: 1>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class B1JointIndex:
    """
    Members:
    
      kHeadYaw
    
      kHeadPitch
    
      kLeftShoulderPitch
    
      kLeftShoulderRoll
    
      kLeftElbowPitch
    
      kLeftElbowYaw
    
      kRightShoulderPitch
    
      kRightShoulderRoll
    
      kRightElbowPitch
    
      kRightElbowYaw
    
      kWaist
    
      kLeftHipPitch
    
      kLeftHipRoll
    
      kLeftHipYaw
    
      kLeftKneePitch
    
      kCrankUpLeft
    
      kCrankDownLeft
    
      kRightHipPitch
    
      kRightHipRoll
    
      kRightHipYaw
    
      kRightKneePitch
    
      kCrankUpRight
    
      kCrankDownRight
    """
    __members__: typing.ClassVar[dict[str, B1JointIndex]]  # value = {'kHeadYaw': <B1JointIndex.kHeadYaw: 0>, 'kHeadPitch': <B1JointIndex.kHeadPitch: 1>, 'kLeftShoulderPitch': <B1JointIndex.kLeftShoulderPitch: 2>, 'kLeftShoulderRoll': <B1JointIndex.kLeftShoulderRoll: 3>, 'kLeftElbowPitch': <B1JointIndex.kLeftElbowPitch: 4>, 'kLeftElbowYaw': <B1JointIndex.kLeftElbowYaw: 5>, 'kRightShoulderPitch': <B1JointIndex.kRightShoulderPitch: 6>, 'kRightShoulderRoll': <B1JointIndex.kRightShoulderRoll: 7>, 'kRightElbowPitch': <B1JointIndex.kRightElbowPitch: 8>, 'kRightElbowYaw': <B1JointIndex.kRightElbowYaw: 9>, 'kWaist': <B1JointIndex.kWaist: 10>, 'kLeftHipPitch': <B1JointIndex.kLeftHipPitch: 11>, 'kLeftHipRoll': <B1JointIndex.kLeftHipRoll: 12>, 'kLeftHipYaw': <B1JointIndex.kLeftHipYaw: 13>, 'kLeftKneePitch': <B1JointIndex.kLeftKneePitch: 14>, 'kCrankUpLeft': <B1JointIndex.kCrankUpLeft: 15>, 'kCrankDownLeft': <B1JointIndex.kCrankDownLeft: 16>, 'kRightHipPitch': <B1JointIndex.kRightHipPitch: 17>, 'kRightHipRoll': <B1JointIndex.kRightHipRoll: 18>, 'kRightHipYaw': <B1JointIndex.kRightHipYaw: 19>, 'kRightKneePitch': <B1JointIndex.kRightKneePitch: 20>, 'kCrankUpRight': <B1JointIndex.kCrankUpRight: 21>, 'kCrankDownRight': <B1JointIndex.kCrankDownRight: 22>}
    kCrankDownLeft: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kCrankDownLeft: 16>
    kCrankDownRight: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kCrankDownRight: 22>
    kCrankUpLeft: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kCrankUpLeft: 15>
    kCrankUpRight: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kCrankUpRight: 21>
    kHeadPitch: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kHeadPitch: 1>
    kHeadYaw: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kHeadYaw: 0>
    kLeftElbowPitch: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kLeftElbowPitch: 4>
    kLeftElbowYaw: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kLeftElbowYaw: 5>
    kLeftHipPitch: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kLeftHipPitch: 11>
    kLeftHipRoll: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kLeftHipRoll: 12>
    kLeftHipYaw: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kLeftHipYaw: 13>
    kLeftKneePitch: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kLeftKneePitch: 14>
    kLeftShoulderPitch: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kLeftShoulderPitch: 2>
    kLeftShoulderRoll: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kLeftShoulderRoll: 3>
    kRightElbowPitch: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kRightElbowPitch: 8>
    kRightElbowYaw: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kRightElbowYaw: 9>
    kRightHipPitch: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kRightHipPitch: 17>
    kRightHipRoll: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kRightHipRoll: 18>
    kRightHipYaw: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kRightHipYaw: 19>
    kRightKneePitch: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kRightKneePitch: 20>
    kRightShoulderPitch: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kRightShoulderPitch: 6>
    kRightShoulderRoll: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kRightShoulderRoll: 7>
    kWaist: typing.ClassVar[B1JointIndex]  # value = <B1JointIndex.kWaist: 10>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class B1LocoApiId:
    """
    Members:
    
      kChangeMode
    
      kMove
    
      kRotateHead
    """
    __members__: typing.ClassVar[dict[str, B1LocoApiId]]  # value = {'kChangeMode': <B1LocoApiId.kChangeMode: 2000>, 'kMove': <B1LocoApiId.kMove: 2001>, 'kRotateHead': <B1LocoApiId.kRotateHead: 2004>}
    kChangeMode: typing.ClassVar[B1LocoApiId]  # value = <B1LocoApiId.kChangeMode: 2000>
    kMove: typing.ClassVar[B1LocoApiId]  # value = <B1LocoApiId.kMove: 2001>
    kRotateHead: typing.ClassVar[B1LocoApiId]  # value = <B1LocoApiId.kRotateHead: 2004>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class B1LocoClient:
    """
    
            B1LocoClient is a client interface for controlling the B1 robot's locomotion and other high-level functionalities.
            It provides methods to send API requests, change robot modes, move the robot, control its head and hands, and more.
            .def("Init", py::overload_cast<const std::string &>(&robot::b1::B1LocoClient::Init), py::arg("robot_name"), R"pbdoc(
                    /**
                     * @brief Initialize the B1LocoClient with a specific robot name.
                     * 
                     * @param robot_name The name of the robot to initialize the client for.
                     */
                
    """
    def ChangeMode(self, mode: RobotMode) -> int:
        """
                        /**
                         * @brief Change robot mode
                         * 
                         * @param mode robot mode, options are: kDamping, kPrepare, kWalking
                         * 
                         * @return 0 if success, otherwise return error code
                         */
        """
    def ControlDexterousHand(self, finger_params: collections.abc.Sequence[DexterousFingerParameter], hand_index: B1HandIndex) -> int:
        """
                        /**
                         * @brief Control dexterous hand
                         *
                         * @param finger_params finger parameters, include position, force, speed, see `DexterousFingerParameter`
                         * @param hand_index hand index, options are: kLeftHand, kRightHand
                         *
                         * @return 0 if success, otherwise return error code
                         */
        """
    def ControlGripper(self, motion_param: GripperMotionParameter, mode: GripperControlMode, hand_index: B1HandIndex) -> int:
        """
                        /**
                         * @brief Control gripper
                         *
                         * @param motion_param motion parameter, include position, force, velocity, see `GripperMotionParameter`
                         * @param mode gripper control mode, options are: kPosition, kForce, see `GripperControlMode`
                         * @param hand_index hand index, options are: kLeftHand, kRightHand
                         *
                         * @return 0 if success, otherwise return error code
                         */
        """
    def GetFrameTransform(self, src: Frame, dst: Frame, transform: Transform) -> int:
        """
                        /**
                         * @brief Get frame transform
                         *
                         * @param src source frame
                         * @param dst destination frame
                         * @param transform [out] calculated transform
                         *
                         * @return 0 if success, otherwise return error code
                         */
        """
    def GetMode(self, get_mode_response: GetModeResponse) -> int:
        """
                        /**
                         * @brief Get current robot mode
                         *
                         * @param[out] get_mode_response Reference to store the response data, including:
                         *              - current_mode (RobotMode enum value)
                         *
                         * @return 0 if success, otherwise return error code
                         * @see ChangeMode() for mode switching API
                         * @see RobotMode enum for available mode definitions
                         */
        """
    def GetUp(self) -> int:
        """
                        /**
                         * @brief Stand up
                         *
                         * @return 0 if success, otherwise return error code
                         */
        """
    def Handshake(self, action: B1HandAction) -> int:
        """
                         /**
                         * @brief Handshake
                         *
                         * @param action whether to start handshake action, options are: kHandOpen, kHandClose
                         *
                         * @return 0 if success, otherwise return error code
                         */
        """
    @typing.overload
    def Init(self) -> None:
        """
        Init
        """
    @typing.overload
    def Init(self, arg0: str) -> None:
        """
        Init with robot name
        """
    def LieDown(self) -> int:
        """
                        /**
                         * @brief Lie down
                         *
                         * @return 0 if success, otherwise return error code
                         */
        """
    def Move(self, vx: typing.SupportsFloat, vy: typing.SupportsFloat, vyaw: typing.SupportsFloat) -> int:
        """
                        /**
                         * @brief Move robot
                         * 
                         * @param vx linear velocity in x direction, unit: m/s
                         * @param vy linear velocity in y direction, unit: m/s
                         * @param vyaw angular velocity, unit: rad/s
                         * 
                         * @return 0 if success, otherwise return error code
                         */
        """
    def MoveHandEndEffector(self, target_posture: Posture, time_millis: typing.SupportsInt, hand_index: B1HandIndex) -> int:
        """
                        /**
                         *  @brief Move hand end-effector with a target posture(position & orientation)
                         *  @deprecated **This API is deprecated and will be removed in future versions.**
                         *              Please use the new API `MoveHandEndEffectorV2` instead.
                         *  @param target_posture Represents the target posture in base frame (torso frame) that the hand end-effector should reach. 
                         *                        It contains position & orientation. 
                         *  @param time_mills Specifies the duration, in milliseconds, for completing the movement.
                         *  @param hand_index Identifies which hand the parameter refers to (for instance, left hand or right hand).
                         *
                         *  @return 0 if success, otherwise return error code
                         * 
                         *  @details
                         *  **Reason for deprecation**: This API is deprecated due to an implicit rotational offset (rot) being applied to the target orientation. 
                         *  The final orientation is calculated as orientation = rot * offset, which contradicts the parameter description of `target_posture`.
                         */
        """
    def MoveHandEndEffectorV2(self, target_posture: Posture, time_millis: typing.SupportsInt, hand_index: B1HandIndex) -> int:
        """
                        /**
                         *  @brief Move hand end-effector with a target posture(position & orientation)
                         *
                         *  @param target_posture Represents the target posture in base frame (torso frame) that the hand end-effector should reach. It contains position & orientation. 
                         *  @param time_mills Specifies the duration, in milliseconds, for completing the movement.
                         *  @param hand_index Identifies which hand the parameter refers to (for instance, left hand or right hand).
                         *
                         *  @return 0 if success, otherwise return error code
                         */
        """
    def MoveHandEndEffectorWithAux(self, target_posture: Posture, aux_posture: Posture, time_millis: typing.SupportsInt, hand_index: B1HandIndex) -> int:
        """
                        /**
                         *  @brief Move hand end-effector to a target posture(position & orientation) with an auxiliary point
                         *
                         *  @param target_posture Represents the target posture in base frame (torso frame) that the hand end-effector should reach. 
                         *  It contains position & orientation.
                         *  @param aux_posture Represents the auxiliary point on the end-effector's motion arc trajectory
                         *  @param time_mills Specifies the duration, in milliseconds, for completing the movement.
                         *  @param hand_index Identifies which hand the parameter refers to (for instance, left hand or right hand).
                         *
                         *  @return 0 if success, otherwise return error code
                         */
        """
    def RotateHead(self, pitch: typing.SupportsFloat, yaw: typing.SupportsFloat) -> int:
        """
                         /**
                         * @brief Robot rotates its head
                         *
                         * @param pitch pitch angle, unit: rad
                         * @param yaw yaw angle, unit: rad
                         *
                         * @return 0 if success, otherwise return error code
                         */
        """
    def RotateHeadWithDirection(self, pitch_direction: typing.SupportsInt, yaw_direction: typing.SupportsInt) -> int:
        """
                         /**
                         * @brief Robot rotates its head with direction
                         *
                         * @param pitch_direction pitch direction, unit: rad
                         * @param yaw_direction yaw direction, unit: rad
                         *
                         * @return 0 if success, otherwise return error code
                         */
        """
    def SendApiRequest(self, api_id: B1LocoApiId, param: str) -> int:
        """
                        /**
                         * @brief Send API request to B1 robot
                         * 
                         * @param api_id API_ID, you can find the API_ID in b1_api_const.hpp
                         * @param param API parameter
                         * 
                         * @return 0 if success, otherwise return error code
                         */
        """
    def SwitchHandEndEffectorControlMode(self, switch_on: bool) -> int:
        """
                        /**
                         * @brief Switch hand end-effector control mode
                         * 
                         * @param switch_on true to switch on, false to switch off
                         * 
                         * @return 0 if success, otherwise return error code
                         */
        """
    def WaveHand(self, action: B1HandAction) -> int:
        """
                         /**
                         * @brief Robot waves hand
                         *
                         * @param action hand action, options are: kHandOpen, kHandClose
                         *
                         * @return 0 if success, otherwise return error code
                         */
        """
    def __init__(self) -> None:
        ...
class B1LowCmdPublisher:
    def CloseChannel(self) -> None:
        """
        Close low cmd publication channel
        """
    def GetChannelName(self) -> str:
        """
        Get low cmd publication channel name
        """
    def InitChannel(self) -> None:
        """
        Init low cmd publication channel
        """
    def Write(self, msg: LowCmd) -> bool:
        """
                         /**
                         * @brief write low cmd message into channel, i.e. publish low cmd message
                         *
                         * @param msg LowCmd
                         *
                         */
        """
    def __init__(self) -> None:
        ...
class B1LowHandDataScriber:
    def CloseChannel(self) -> None:
        """
        Close low state subscription channel
        """
    def GetChannelName(self) -> str:
        """
        Get low state subscription channel name
        """
    def InitChannel(self) -> None:
        """
        Init low state subscription channel
        """
    def __init__(self, handler: collections.abc.Callable) -> None:
        """
                         /**
                         * @brief init hand data subscriber with callback handler
                         *
                         * @param handler callback handler of hand data, the handler should accept one parameter of type LowState
                         *
                         */
        """
class B1LowStateSubscriber:
    def CloseChannel(self) -> None:
        """
        Close low state subscription channel
        """
    def GetChannelName(self) -> str:
        """
        Get low state subscription channel name
        """
    def InitChannel(self) -> None:
        """
        Init low state subscription channel
        """
    def __init__(self, handler: collections.abc.Callable) -> None:
        """
                         /**
                         * @brief init low state subscriber with callback handler
                         *
                         * @param handler callback handler of low state, the handler should accept one parameter of type LowState
                         *
                         */
        """
class B1OdometerStateSubscriber:
    def CloseChannel(self) -> None:
        """
        Close odometer subscription channel
        """
    def GetChannelName(self) -> str:
        """
        Get odometer subscription channel name
        """
    def InitChannel(self) -> None:
        """
        Init odometer subscription channel
        """
    def __init__(self, handler: collections.abc.Callable) -> None:
        """
                         /**
                         * @brief init odometer state subscriber with callback handler
                         *
                         * @param handler callback handler of odom state, the handler should accept one parameter of type Odometer
                         *
                         */
        """
class ChannelFactory:
    @staticmethod
    def Instance() -> ChannelFactory:
        """
                                Get the singleton instance of the channel factory.
        
                                Note: The returned instance is managed internally and should not be deleted or modified.
        """
    def Init(self, domain_id: typing.SupportsInt, network_interface: str = '') -> None:
        """
                        domain_id: domain id of DDS
                        network_interface: network interface of DDS, default empty string
        """
class DexterousFingerParameter:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, seq: typing.SupportsInt, angle: typing.SupportsInt, force: typing.SupportsInt, speed: typing.SupportsInt) -> None:
        ...
    @property
    def angle(self) -> int:
        ...
    @angle.setter
    def angle(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def force(self) -> int:
        ...
    @force.setter
    def force(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def seq(self) -> int:
        ...
    @seq.setter
    def seq(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def speed(self) -> int:
        ...
    @speed.setter
    def speed(self, arg0: typing.SupportsInt) -> None:
        ...
class Frame:
    """
    Members:
    
      kUnknown
    
      kBody
    
      kHead
    
      kLeftHand
    
      kRightHand
    
      kLeftFoot
    
      kRightFoot
    """
    __members__: typing.ClassVar[dict[str, Frame]]  # value = {'kUnknown': <Frame.kUnknown: -1>, 'kBody': <Frame.kBody: 0>, 'kHead': <Frame.kHead: 1>, 'kLeftHand': <Frame.kLeftHand: 2>, 'kRightHand': <Frame.kRightHand: 3>, 'kLeftFoot': <Frame.kLeftFoot: 4>, 'kRightFoot': <Frame.kRightFoot: 5>}
    kBody: typing.ClassVar[Frame]  # value = <Frame.kBody: 0>
    kHead: typing.ClassVar[Frame]  # value = <Frame.kHead: 1>
    kLeftFoot: typing.ClassVar[Frame]  # value = <Frame.kLeftFoot: 4>
    kLeftHand: typing.ClassVar[Frame]  # value = <Frame.kLeftHand: 2>
    kRightFoot: typing.ClassVar[Frame]  # value = <Frame.kRightFoot: 5>
    kRightHand: typing.ClassVar[Frame]  # value = <Frame.kRightHand: 3>
    kUnknown: typing.ClassVar[Frame]  # value = <Frame.kUnknown: -1>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class GetModeResponse:
    mode: RobotMode
    def __init__(self) -> None:
        ...
class GripperControlMode:
    """
    Members:
    
      kPosition : Position mode: stops at target position or specified reaction force
    
      kForce : Force mode: continues to move with specified force if target position is not reached
    """
    __members__: typing.ClassVar[dict[str, GripperControlMode]]  # value = {'kPosition': <GripperControlMode.kPosition: 0>, 'kForce': <GripperControlMode.kForce: 1>}
    kForce: typing.ClassVar[GripperControlMode]  # value = <GripperControlMode.kForce: 1>
    kPosition: typing.ClassVar[GripperControlMode]  # value = <GripperControlMode.kPosition: 0>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class GripperMotionParameter:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, position: typing.SupportsInt, force: typing.SupportsInt, speed: typing.SupportsInt) -> None:
        ...
    @property
    def force(self) -> int:
        ...
    @force.setter
    def force(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def position(self) -> int:
        ...
    @position.setter
    def position(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def speed(self) -> int:
        ...
    @speed.setter
    def speed(self, arg0: typing.SupportsInt) -> None:
        ...
class HandReplyData:
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: HandReplyData) -> bool:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: HandReplyData) -> None:
        ...
    def __ne__(self, arg0: HandReplyData) -> bool:
        ...
    @property
    def hand_data(self) -> list[HandReplyParam]:
        ...
    @hand_data.setter
    def hand_data(self, arg1: collections.abc.Sequence[HandReplyParam]) -> None:
        ...
    @property
    def hand_index(self) -> int:
        ...
    @hand_index.setter
    def hand_index(self) -> int:
        ...
    @property
    def hand_type(self) -> int:
        ...
    @hand_type.setter
    def hand_type(self) -> int:
        ...
class HandReplyParam:
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: HandReplyParam) -> bool:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: HandReplyParam) -> None:
        ...
    def __ne__(self, arg0: HandReplyParam) -> bool:
        ...
    @property
    def angle(self) -> int:
        ...
    @angle.setter
    def angle(self) -> int:
        ...
    @property
    def current(self) -> int:
        ...
    @current.setter
    def current(self) -> int:
        ...
    @property
    def error(self) -> int:
        ...
    @error.setter
    def error(self) -> int:
        ...
    @property
    def force(self) -> int:
        ...
    @force.setter
    def force(self) -> int:
        ...
    @property
    def seq(self) -> int:
        ...
    @seq.setter
    def seq(self) -> int:
        ...
    @property
    def status(self) -> int:
        ...
    @status.setter
    def status(self) -> int:
        ...
    @property
    def temp(self) -> int:
        ...
    @temp.setter
    def temp(self) -> int:
        ...
class ImuState:
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: ImuState) -> bool:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: ImuState) -> None:
        ...
    def __ne__(self, arg0: ImuState) -> bool:
        ...
    @property
    def acc(self) -> typing.Annotated[list[float], "FixedSize(3)"]:
        ...
    @acc.setter
    def acc(self, arg1: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"]) -> None:
        ...
    @property
    def gyro(self) -> typing.Annotated[list[float], "FixedSize(3)"]:
        ...
    @gyro.setter
    def gyro(self, arg1: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"]) -> None:
        ...
    @property
    def rpy(self) -> typing.Annotated[list[float], "FixedSize(3)"]:
        ...
    @rpy.setter
    def rpy(self, arg1: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"]) -> None:
        ...
class LowCmd:
    __hash__: typing.ClassVar[None] = None
    cmd_type: LowCmdType
    def __eq__(self, arg0: LowCmd) -> bool:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: LowCmd) -> None:
        ...
    def __ne__(self, arg0: LowCmd) -> bool:
        ...
    @property
    def motor_cmd(self) -> list[MotorCmd]:
        ...
    @motor_cmd.setter
    def motor_cmd(self, arg1: collections.abc.Sequence[MotorCmd]) -> None:
        ...
class LowCmdType:
    """
    Members:
    
      PARALLEL
    
      SERIAL
    """
    PARALLEL: typing.ClassVar[LowCmdType]  # value = <LowCmdType.PARALLEL: 0>
    SERIAL: typing.ClassVar[LowCmdType]  # value = <LowCmdType.SERIAL: 1>
    __members__: typing.ClassVar[dict[str, LowCmdType]]  # value = {'PARALLEL': <LowCmdType.PARALLEL: 0>, 'SERIAL': <LowCmdType.SERIAL: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class LowState:
    __hash__: typing.ClassVar[None] = None
    imu_state: ImuState
    def __eq__(self, arg0: LowState) -> bool:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: LowState) -> None:
        ...
    def __ne__(self, arg0: LowState) -> bool:
        ...
    @property
    def motor_state_parallel(self) -> list[MotorState]:
        ...
    @motor_state_parallel.setter
    def motor_state_parallel(self, arg1: collections.abc.Sequence[MotorState]) -> None:
        ...
    @property
    def motor_state_serial(self) -> list[MotorState]:
        ...
    @motor_state_serial.setter
    def motor_state_serial(self, arg1: collections.abc.Sequence[MotorState]) -> None:
        ...
class MotorCmd:
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: MotorCmd) -> bool:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: MotorCmd) -> None:
        ...
    def __ne__(self, arg0: MotorCmd) -> bool:
        ...
    @property
    def dq(self) -> float:
        ...
    @dq.setter
    def dq(self, arg1: typing.SupportsFloat) -> None:
        ...
    @property
    def kd(self) -> float:
        ...
    @kd.setter
    def kd(self, arg1: typing.SupportsFloat) -> None:
        ...
    @property
    def kp(self) -> float:
        ...
    @kp.setter
    def kp(self, arg1: typing.SupportsFloat) -> None:
        ...
    @property
    def mode(self) -> int:
        ...
    @mode.setter
    def mode(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def q(self) -> float:
        ...
    @q.setter
    def q(self, arg1: typing.SupportsFloat) -> None:
        ...
    @property
    def tau(self) -> float:
        ...
    @tau.setter
    def tau(self, arg1: typing.SupportsFloat) -> None:
        ...
    @property
    def weight(self) -> float:
        ...
    @weight.setter
    def weight(self, arg1: typing.SupportsFloat) -> None:
        ...
class MotorState:
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: MotorState) -> bool:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: MotorState) -> None:
        ...
    def __ne__(self, arg0: MotorState) -> bool:
        ...
    @property
    def ddq(self) -> float:
        ...
    @ddq.setter
    def ddq(self) -> float:
        ...
    @property
    def dq(self) -> float:
        ...
    @dq.setter
    def dq(self) -> float:
        ...
    @property
    def lost(self) -> int:
        ...
    @lost.setter
    def lost(self) -> int:
        ...
    @property
    def mode(self) -> int:
        ...
    @mode.setter
    def mode(self) -> int:
        ...
    @property
    def q(self) -> float:
        ...
    @q.setter
    def q(self) -> float:
        ...
    @property
    def reserve(self) -> typing.Annotated[list[int], "FixedSize(2)"]:
        ...
    @reserve.setter
    def reserve(self) -> typing.Annotated[list[int], "FixedSize(2)"]:
        ...
    @property
    def tau_est(self) -> float:
        ...
    @tau_est.setter
    def tau_est(self) -> float:
        ...
    @property
    def temperature(self) -> int:
        ...
    @temperature.setter
    def temperature(self) -> int:
        ...
class Odometer:
    def __init__(self) -> None:
        ...
    @property
    def theta(self) -> float:
        ...
    @theta.setter
    def theta(self, arg1: typing.SupportsFloat) -> None:
        ...
    @property
    def x(self) -> float:
        ...
    @x.setter
    def x(self, arg1: typing.SupportsFloat) -> None:
        ...
    @property
    def y(self) -> float:
        ...
    @y.setter
    def y(self, arg1: typing.SupportsFloat) -> None:
        ...
class Orientation:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, roll: typing.SupportsFloat, pitch: typing.SupportsFloat, yaw: typing.SupportsFloat) -> None:
        ...
    @property
    def pitch(self) -> float:
        ...
    @pitch.setter
    def pitch(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def roll(self) -> float:
        ...
    @roll.setter
    def roll(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def yaw(self) -> float:
        ...
    @yaw.setter
    def yaw(self, arg0: typing.SupportsFloat) -> None:
        ...
class Position:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, x: typing.SupportsFloat, y: typing.SupportsFloat, z: typing.SupportsFloat) -> None:
        ...
    @property
    def x(self) -> float:
        ...
    @x.setter
    def x(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def y(self) -> float:
        ...
    @y.setter
    def y(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def z(self) -> float:
        ...
    @z.setter
    def z(self, arg0: typing.SupportsFloat) -> None:
        ...
class Posture:
    orientation: Orientation
    position: Position
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, position: Position, orientation: Orientation) -> None:
        ...
class Quaternion:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, x: typing.SupportsFloat, y: typing.SupportsFloat, z: typing.SupportsFloat, w: typing.SupportsFloat) -> None:
        ...
    @property
    def w(self) -> float:
        ...
    @w.setter
    def w(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def x(self) -> float:
        ...
    @x.setter
    def x(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def y(self) -> float:
        ...
    @y.setter
    def y(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def z(self) -> float:
        ...
    @z.setter
    def z(self, arg0: typing.SupportsFloat) -> None:
        ...
class RobotMode:
    """
    Members:
    
      kUnknown
    
      kDamping
    
      kPrepare
    
      kWalking
    
      kCustom
    """
    __members__: typing.ClassVar[dict[str, RobotMode]]  # value = {'kUnknown': <RobotMode.kUnknown: -1>, 'kDamping': <RobotMode.kDamping: 0>, 'kPrepare': <RobotMode.kPrepare: 1>, 'kWalking': <RobotMode.kWalking: 2>, 'kCustom': <RobotMode.kCustom: 3>}
    kCustom: typing.ClassVar[RobotMode]  # value = <RobotMode.kCustom: 3>
    kDamping: typing.ClassVar[RobotMode]  # value = <RobotMode.kDamping: 0>
    kPrepare: typing.ClassVar[RobotMode]  # value = <RobotMode.kPrepare: 1>
    kUnknown: typing.ClassVar[RobotMode]  # value = <RobotMode.kUnknown: -1>
    kWalking: typing.ClassVar[RobotMode]  # value = <RobotMode.kWalking: 2>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Transform:
    orientation: Quaternion
    position: Position
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, position: Position, orientation: Quaternion) -> None:
        ...
B1JointCnt: int = 23
PARALLEL: LowCmdType  # value = <LowCmdType.PARALLEL: 0>
SERIAL: LowCmdType  # value = <LowCmdType.SERIAL: 1>
__version__: str = '0.1'
kBody: Frame  # value = <Frame.kBody: 0>
kChangeMode: B1LocoApiId  # value = <B1LocoApiId.kChangeMode: 2000>
kCrankDownLeft: B1JointIndex  # value = <B1JointIndex.kCrankDownLeft: 16>
kCrankDownRight: B1JointIndex  # value = <B1JointIndex.kCrankDownRight: 22>
kCrankUpLeft: B1JointIndex  # value = <B1JointIndex.kCrankUpLeft: 15>
kCrankUpRight: B1JointIndex  # value = <B1JointIndex.kCrankUpRight: 21>
kCustom: RobotMode  # value = <RobotMode.kCustom: 3>
kDamping: RobotMode  # value = <RobotMode.kDamping: 0>
kForce: GripperControlMode  # value = <GripperControlMode.kForce: 1>
kHandClose: B1HandAction  # value = <B1HandAction.kHandClose: 1>
kHandOpen: B1HandAction  # value = <B1HandAction.kHandOpen: 0>
kHead: Frame  # value = <Frame.kHead: 1>
kHeadPitch: B1JointIndex  # value = <B1JointIndex.kHeadPitch: 1>
kHeadYaw: B1JointIndex  # value = <B1JointIndex.kHeadYaw: 0>
kLeftElbowPitch: B1JointIndex  # value = <B1JointIndex.kLeftElbowPitch: 4>
kLeftElbowYaw: B1JointIndex  # value = <B1JointIndex.kLeftElbowYaw: 5>
kLeftFoot: Frame  # value = <Frame.kLeftFoot: 4>
kLeftHand: Frame  # value = <Frame.kLeftHand: 2>
kLeftHipPitch: B1JointIndex  # value = <B1JointIndex.kLeftHipPitch: 11>
kLeftHipRoll: B1JointIndex  # value = <B1JointIndex.kLeftHipRoll: 12>
kLeftHipYaw: B1JointIndex  # value = <B1JointIndex.kLeftHipYaw: 13>
kLeftKneePitch: B1JointIndex  # value = <B1JointIndex.kLeftKneePitch: 14>
kLeftShoulderPitch: B1JointIndex  # value = <B1JointIndex.kLeftShoulderPitch: 2>
kLeftShoulderRoll: B1JointIndex  # value = <B1JointIndex.kLeftShoulderRoll: 3>
kMove: B1LocoApiId  # value = <B1LocoApiId.kMove: 2001>
kPosition: GripperControlMode  # value = <GripperControlMode.kPosition: 0>
kPrepare: RobotMode  # value = <RobotMode.kPrepare: 1>
kRightElbowPitch: B1JointIndex  # value = <B1JointIndex.kRightElbowPitch: 8>
kRightElbowYaw: B1JointIndex  # value = <B1JointIndex.kRightElbowYaw: 9>
kRightFoot: Frame  # value = <Frame.kRightFoot: 5>
kRightHand: Frame  # value = <Frame.kRightHand: 3>
kRightHipPitch: B1JointIndex  # value = <B1JointIndex.kRightHipPitch: 17>
kRightHipRoll: B1JointIndex  # value = <B1JointIndex.kRightHipRoll: 18>
kRightHipYaw: B1JointIndex  # value = <B1JointIndex.kRightHipYaw: 19>
kRightKneePitch: B1JointIndex  # value = <B1JointIndex.kRightKneePitch: 20>
kRightShoulderPitch: B1JointIndex  # value = <B1JointIndex.kRightShoulderPitch: 6>
kRightShoulderRoll: B1JointIndex  # value = <B1JointIndex.kRightShoulderRoll: 7>
kRotateHead: B1LocoApiId  # value = <B1LocoApiId.kRotateHead: 2004>
kUnknown: Frame  # value = <Frame.kUnknown: -1>
kWaist: B1JointIndex  # value = <B1JointIndex.kWaist: 10>
kWalking: RobotMode  # value = <RobotMode.kWalking: 2>
