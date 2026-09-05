/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    main.c
 * @brief   Active camera, robot-arm, and pneumatic integrated firmware
 *
 * This file preserves the team's verified servo channels, directions,
 * safe home pose (0/90/90/90), and loading pose (90/120/56/0).
 ******************************************************************************
 */
/* USER CODE END Header */

#include "main.h"
#include <stdlib.h>
#include <string.h>

COM_InitTypeDef BspCOMInit;
TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim16;
TIM_HandleTypeDef htim17;

/* USER CODE BEGIN PV */
/* Angles remain expressed in real degrees. */
float current_ch1 = 0.0f;
float current_ch2 = 90.0f;
float current_ch3 = 90.0f;
float current_ch4 = 90.0f;
float right_ch1 = 0.0f;
float right_ch2 = 90.0f;
float right_ch3 = 90.0f;
float right_ch4 = 90.0f;
extern UART_HandleTypeDef hcom_uart[COMn];

typedef enum
{
  COMMAND_NONE = 0,
  COMMAND_STOP,
  COMMAND_HOME,
  COMMAND_LOADING
} CommandOverride;

volatile uint8_t motion_paused = 0U;
volatile uint8_t start_requested = 0U;
volatile uint8_t motion_busy = 0U;
volatile CommandOverride command_override = COMMAND_NONE;
volatile uint8_t emergency_stop_requested = 0U;

typedef struct
{
  float size_mm;
  float j1;
  float approach[3];
  float contact[3];
  float preload[3];
  float lift[3];
} CameraPickPlan;

volatile uint8_t camera_plan_ready = 0U;
volatile uint8_t dual_plan_ready = 0U;
uint8_t dual_left_present = 0U;
uint8_t dual_right_present = 0U;
uint8_t dual_suction_enabled = 1U;
uint8_t dual_pipeline_requested = 1U;
/* 0=none, 1=left arm is holding at loading, 2=right arm is holding. */
uint8_t pipeline_holding_arm = 0U;
float pipeline_holding_size_mm = 0.0f;
volatile uint8_t finish_requested = 0U;
CameraPickPlan camera_plan;
CameraPickPlan dual_left_plan;
CameraPickPlan dual_right_plan;
static char rx_line[384];
static uint16_t rx_length = 0U;
/* USER CODE END PV */

void SystemClock_Config(void);
void PeriphCommonClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM1_Init(void);
static void MX_TIM2_Init(void);
static void MX_TIM16_Init(void);
static void MX_TIM17_Init(void);

/* USER CODE BEGIN PFP */
uint32_t Degree_To_Pulse(float angle);
uint32_t Degree_To_Channel_Pulse(uint32_t channel, float logical_angle);
void Move_Single_Channel_Smooth(uint32_t channel, float *current_angle,
                                float target_angle, uint32_t speed_ms);
void Move_Dual_Channel_Smooth(uint32_t ch_A, float *curr_A, float target_A,
                              uint32_t ch_B, float *curr_B, float target_B,
                              uint32_t speed_ms);
void Move_Triple_Channel_Smooth(uint32_t ch_A, float *curr_A, float target_A,
                                uint32_t ch_B, float *curr_B, float target_B,
                                uint32_t ch_C, float *curr_C, float target_C,
                                uint32_t speed_ms);
void Console_Write(const char *text);
void Left_Suction_Set(uint8_t on);
void Right_Suction_Set(uint8_t on);
void Console_Print_Guide(void);
void Service_Console(void);
void Controlled_Delay(uint32_t delay_ms);
void Go_Home(void);
void Go_Loading(void);
void Run_Pick_Cycle(void);
void Run_Camera_Plan(void);
void Run_Dual_Camera_Plans(void);
void Run_Pipeline_Camera_Plans(void);
void Move_Both_To_Loading_Safely(void);
void Move_Both_To_Home_Safely(void);
/* USER CODE END PFP */

/* USER CODE BEGIN 0 */
#define SERVO_MIN_ANGLE_DEG  0.0f
#define SERVO_MAX_ANGLE_DEG  180.0f
#define SERVO_STEP_DEG       1.0f
#define SERVO_MOVE_DELAY_MS       7U
#define EMPTY_ARM_MOVE_DELAY_MS   5U

void Left_Suction_Set(uint8_t on)
{
  /* CH4 PE4, Low Active: RESET=ON, SET=OFF */
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_4,
                    (on != 0U) ? GPIO_PIN_RESET : GPIO_PIN_SET);
}

void Right_Suction_Set(uint8_t on)
{
  /* CH1 PB11, Low Active: RESET=ON, SET=OFF */
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_11,
                    (on != 0U) ? GPIO_PIN_RESET : GPIO_PIN_SET);
}

void Console_Write(const char *text)
{
  HAL_UART_Transmit(&hcom_uart[COM1], (uint8_t *)text,
                    (uint16_t)strlen(text), 100U);
}

void Console_Print_Guide(void)
{
  Console_Write("\r\n========================================\r\n");
  Console_Write(" 로봇팔 키보드 조작 안내 (115200 baud)\r\n");
  Console_Write(" q : 카메라 통합 모드에서는 사용하지 않음\r\n");
  Console_Write(" w : 일시정지 / 다시 누르면 재개\r\n");
  Console_Write(" e : 안전 기본 자세(0/90/90/90)로 이동\r\n");
  Console_Write(" r : 로딩 자세(90/120/56/0)로 이동\r\n");
  Console_Write(" P,... : PC 카메라 프로그램의 동적 피킹 명령\r\n");
  Console_Write(" 영문 소문자 한 글자를 입력하세요.\r\n");
  Console_Write("========================================\r\n");
}

static void Handle_Key(uint8_t key)
{
  if ((key == '\r') || (key == '\n')) return;

  if ((key == 'q') || (key == 'Q'))
  {
    Console_Write("[안내] 카메라 화면에서 p를 눌러 동적 좌표를 전송하세요.\r\n");
  }
  else if ((key == 'x') || (key == 'X'))
  {
    motion_paused = 0U;
    start_requested = 0U;
    camera_plan_ready = 0U;
    dual_plan_ready = 0U;
    finish_requested = 0U;
    Left_Suction_Set(0U);
    Right_Suction_Set(0U);
    command_override = COMMAND_STOP;
    Console_Write("EMERGENCY_STOP\r\n");
  }
  else if ((key == 'h') || (key == 'H'))
  {
    motion_paused = 0U;
    start_requested = 0U;
    camera_plan_ready = 0U;
    dual_plan_ready = 0U;
    finish_requested = 0U;
    Left_Suction_Set(0U);
    Right_Suction_Set(0U);
    emergency_stop_requested = 1U;
    command_override = COMMAND_HOME;
    Console_Write("RESET_HOME\r\n");
  }
  else if ((key == 'w') || (key == 'W'))
  {
    motion_paused = (motion_paused == 0U) ? 1U : 0U;
    Console_Write((motion_paused != 0U) ?
                  "[입력 w] 일시정지했습니다. 다시 w를 누르면 재개합니다.\r\n" :
                  "[입력 w] 일시정지를 해제하고 동작을 재개합니다.\r\n");
  }
  else if ((key == 'e') || (key == 'E'))
  {
    motion_paused = 0U;
    start_requested = 0U;
    command_override = COMMAND_HOME;
    Console_Write("[입력 e] 현재 동작을 중단했습니다.\r\n");
    Console_Write("[대기] 2초 후 안전 기본 자세(0/90/90/90)로 이동합니다.\r\n");
  }
  else if ((key == 'r') || (key == 'R'))
  {
    motion_paused = 0U;
    start_requested = 0U;
    command_override = COMMAND_LOADING;
    Console_Write("[입력 r] 현재 동작을 중단했습니다.\r\n");
    Console_Write("[대기] 2초 후 로딩 자세로 이동합니다.\r\n");
  }
  else
  {
    Console_Write("[안내] 사용할 수 없는 키입니다. q, w, e, r 중 하나를 입력하세요.\r\n");
  }
}

static uint8_t Plan_Values_Valid(const float *values)
{
  uint32_t i;

  if ((values[0] < 20.0f) || (values[0] > 60.0f)) return 0U;
  for (i = 1U; i < 14U; i++)
  {
    if ((values[i] < 0.0f) || (values[i] > 180.0f)) return 0U;
  }
  if ((values[3] > 90.0f) || (values[6] > 90.0f) ||
      (values[9] > 90.0f) || (values[12] > 90.0f)) return 0U;
  return 1U;
}

static void Plan_From_Values(CameraPickPlan *plan, const float *values)
{
  plan->size_mm = values[0];
  plan->j1 = values[1];
  plan->approach[0] = values[2];
  plan->approach[1] = values[3];
  plan->approach[2] = values[4];
  plan->contact[0] = values[5];
  plan->contact[1] = values[6];
  plan->contact[2] = values[7];
  plan->preload[0] = values[8];
  plan->preload[1] = values[9];
  plan->preload[2] = values[10];
  plan->lift[0] = values[11];
  plan->lift[1] = values[12];
  plan->lift[2] = values[13];
}

static void Process_Line(char *line)
{
  CameraPickPlan next;
  float values[30];
  char *token;
  char *end;
  uint32_t i;

  if ((line[0] != '\0') && (line[1] == '\0'))
  {
    if ((line[0] == 'F') || (line[0] == 'f'))
    {
      if (motion_busy != 0U)
        Console_Write("BUSY\r\n");
      else
      {
        finish_requested = 1U;
        Console_Write("FINISH_OK\r\n");
      }
      return;
    }
    Handle_Key((uint8_t)line[0]);
    return;
  }

  if (((line[0] == 'D') || (line[0] == 'M')) && (line[1] == ','))
  {
    uint8_t pipeline_command = (line[0] == 'D') ? 1U : 0U;
    token = strtok(&line[2], ",");
    for (i = 0U; i < 30U; i++)
    {
      if (token == NULL)
      {
        Console_Write("ERROR\r\n");
        return;
      }
      values[i] = strtof(token, &end);
      if ((end == token) || (*end != '\0'))
      {
        Console_Write("ERROR\r\n");
        return;
      }
      token = strtok(NULL, ",");
    }
    if ((token != NULL) ||
        !((values[0] == 0.0f) || (values[0] == 1.0f)) ||
        !((values[15] == 0.0f) || (values[15] == 1.0f)) ||
        ((values[0] != 0.0f) && (Plan_Values_Valid(&values[1]) == 0U)) ||
        ((values[15] != 0.0f) && (Plan_Values_Valid(&values[16]) == 0U)))
    {
      Console_Write("ERROR\r\n");
      return;
    }
    if (motion_busy != 0U)
    {
      Console_Write("BUSY\r\n");
      return;
    }
    dual_left_present = (values[0] != 0.0f) ? 1U : 0U;
    dual_right_present = (values[15] != 0.0f) ? 1U : 0U;
    dual_suction_enabled = 1U;
    if (dual_left_present != 0U) Plan_From_Values(&dual_left_plan, &values[1]);
    if (dual_right_present != 0U) Plan_From_Values(&dual_right_plan, &values[16]);
    /*
     * 교차 순차제어:
     * - 보유 물체 없음: 한 팔 계획만 받아 집은 뒤 전체 로딩에서 유지
     * - 왼팔 보유: 오른팔 계획과 교차(왼팔 놓기/오른팔 집기), 또는 0/0으로 놓기만
     * - 오른팔 보유: 위 동작의 좌우 반대
     */
    if (((pipeline_command == 0U) &&
         ((dual_left_present + dual_right_present) == 0U)) ||
        ((pipeline_command != 0U) &&
         (((pipeline_holding_arm == 0U) &&
           ((dual_left_present + dual_right_present) != 1U)) ||
          ((pipeline_holding_arm == 1U) && (dual_left_present != 0U)) ||
          ((pipeline_holding_arm == 2U) && (dual_right_present != 0U)) ||
          ((dual_left_present != 0U) && (dual_right_present != 0U)))))
    {
      Console_Write("ERROR\r\n");
      return;
    }
    dual_pipeline_requested = pipeline_command;
    dual_plan_ready = 1U;
    Console_Write("PLAN_OK\r\n");
    return;
  }

  /*
   * T,<14 plan values>
   * 오른팔만 전체 피킹 경로를 시험한다. 공압 CH1은 사용하지 않는다.
   */
  if ((line[0] == 'T') && (line[1] == ','))
  {
    token = strtok(&line[2], ",");
    for (i = 0U; i < 14U; i++)
    {
      if (token == NULL)
      {
        Console_Write("ERROR\r\n");
        return;
      }
      values[i] = strtof(token, &end);
      if ((end == token) || (*end != '\0'))
      {
        Console_Write("ERROR\r\n");
        return;
      }
      token = strtok(NULL, ",");
    }
    if ((token != NULL) || (Plan_Values_Valid(values) == 0U))
    {
      Console_Write("ERROR\r\n");
      return;
    }
    if (motion_busy != 0U)
    {
      Console_Write("BUSY\r\n");
      return;
    }
    dual_left_present = 0U;
    dual_right_present = 1U;
    dual_suction_enabled = 0U;
    Plan_From_Values(&dual_right_plan, values);
    dual_plan_ready = 1U;
    Console_Write("PLAN_OK\r\n");
    return;
  }

  if ((line[0] != 'P') || (line[1] != ','))
  {
    Console_Write("ERROR\r\n");
    return;
  }
  token = strtok(&line[2], ",");
  for (i = 0U; i < 14U; i++)
  {
    if (token == NULL)
    {
      Console_Write("ERROR\r\n");
      return;
    }
    values[i] = strtof(token, &end);
    if ((end == token) || (*end != '\0'))
    {
      Console_Write("ERROR\r\n");
      return;
    }
    token = strtok(NULL, ",");
  }
  if (token != NULL)
  {
    Console_Write("ERROR\r\n");
    return;
  }
  if (Plan_Values_Valid(values) == 0U)
  {
    Console_Write("ERROR\r\n");
    return;
  }

  Plan_From_Values(&next, values);
  if (motion_busy != 0U)
  {
    Console_Write("BUSY\r\n");
    return;
  }
  camera_plan = next;
  camera_plan_ready = 1U;
  Console_Write("PLAN_OK\r\n");
}

void Service_Console(void)
{
  uint8_t key;

  /* A camera plan is a burst of about 70-100 bytes. A zero timeout can read
     one byte and then lose the rest to UART overrun, so allow 2 ms per byte. */
  while (HAL_UART_Receive(&hcom_uart[COM1], &key, 1U, 2U) == HAL_OK)
  {
    if ((key == '\r') || (key == '\n'))
    {
      if (rx_length > 0U)
      {
        rx_line[rx_length] = '\0';
        Process_Line(rx_line);
        rx_length = 0U;
      }
    }
    else if (rx_length < (sizeof(rx_line) - 1U))
    {
      rx_line[rx_length++] = (char)key;
    }
    else
    {
      rx_length = 0U;
      Console_Write("ERROR\r\n");
    }
  }

  while ((motion_paused != 0U) && (command_override == COMMAND_NONE))
  {
    if (HAL_UART_Receive(&hcom_uart[COM1], &key, 1U, 10U) == HAL_OK)
    {
      if ((key == 'w') || (key == 'W') || (key == 'e') || (key == 'E') ||
          (key == 'r') || (key == 'R'))
        Handle_Key(key);
    }
    HAL_Delay(10U);
  }
}

void Controlled_Delay(uint32_t delay_ms)
{
  uint32_t elapsed = 0U;
  while ((elapsed < delay_ms) && (command_override == COMMAND_NONE))
  {
    Service_Console();
    HAL_Delay(10U);
    elapsed += 10U;
  }
}

static float Clamp_Angle(float angle)
{
  if (angle < SERVO_MIN_ANGLE_DEG) return SERVO_MIN_ANGLE_DEG;
  if (angle > SERVO_MAX_ANGLE_DEG) return SERVO_MAX_ANGLE_DEG;
  return angle;
}

static float Abs_Float(float value)
{
  return (value < 0.0f) ? -value : value;
}

static float Smooth_Progress(float t)
{
  /* Slow start and slow stop, like a simulator animation. */
  return t * t * (3.0f - 2.0f * t);
}

static uint32_t Steps_For_Delta(float delta)
{
  uint32_t steps = (uint32_t)(Abs_Float(delta) / SERVO_STEP_DEG + 0.999f);
  return (steps == 0U) ? 1U : steps;
}

uint32_t Degree_To_Pulse(float angle)
{
  angle = Clamp_Angle(angle);
  /* Preserve the verified mapping: 0 deg=500 us, 90 deg=1500 us,
     180 deg=2500 us. Add 0.5 before conversion for nearest-integer rounding. */
  return (uint32_t)(500.0f + (angle * 2000.0f / 180.0f) + 0.5f);
}

uint32_t Degree_To_Channel_Pulse(uint32_t channel, float logical_angle)
{
  (void)channel;
  return Degree_To_Pulse(logical_angle);
}

void Move_Single_Channel_Smooth(uint32_t channel, float *current_angle,
                                float target_angle, uint32_t speed_ms)
{
  float start_angle;
  float delta;
  uint32_t steps;
  uint32_t i;

  target_angle = Clamp_Angle(target_angle);
  start_angle = *current_angle;
  delta = target_angle - start_angle;
  steps = Steps_For_Delta(delta);

  for (i = 1U; i <= steps; i++)
  {
    Service_Console();
    if (command_override != COMMAND_NONE) return;
    float t = (float)i / (float)steps;
    float progress = Smooth_Progress(t);
    *current_angle = start_angle + delta * progress;
    __HAL_TIM_SET_COMPARE(&htim2, channel, Degree_To_Channel_Pulse(channel, *current_angle));
    Controlled_Delay(speed_ms);
  }
  *current_angle = target_angle;
  __HAL_TIM_SET_COMPARE(&htim2, channel, Degree_To_Channel_Pulse(channel, target_angle));
}

void Move_Dual_Channel_Smooth(uint32_t ch_A, float *curr_A, float target_A,
                              uint32_t ch_B, float *curr_B, float target_B,
                              uint32_t speed_ms)
{
  float start_A, start_B, delta_A, delta_B, max_delta;
  uint32_t steps, i;

  target_A = Clamp_Angle(target_A);
  target_B = Clamp_Angle(target_B);
  start_A = *curr_A;
  start_B = *curr_B;
  delta_A = target_A - start_A;
  delta_B = target_B - start_B;
  max_delta = Abs_Float(delta_A);
  if (Abs_Float(delta_B) > max_delta) max_delta = Abs_Float(delta_B);
  steps = Steps_For_Delta(max_delta);

  for (i = 1U; i <= steps; i++)
  {
    Service_Console();
    if (command_override != COMMAND_NONE) return;
    float t = (float)i / (float)steps;
    float progress = Smooth_Progress(t);
    *curr_A = start_A + delta_A * progress;
    *curr_B = start_B + delta_B * progress;

    __HAL_TIM_SET_COMPARE(&htim2, ch_A, Degree_To_Channel_Pulse(ch_A, *curr_A));
    __HAL_TIM_SET_COMPARE(&htim2, ch_B, Degree_To_Channel_Pulse(ch_B, *curr_B));
    Controlled_Delay(speed_ms);
  }
  *curr_A = target_A;
  *curr_B = target_B;
  __HAL_TIM_SET_COMPARE(&htim2, ch_A, Degree_To_Channel_Pulse(ch_A, target_A));
  __HAL_TIM_SET_COMPARE(&htim2, ch_B, Degree_To_Channel_Pulse(ch_B, target_B));
}

void Move_Triple_Channel_Smooth(uint32_t ch_A, float *curr_A, float target_A,
                                uint32_t ch_B, float *curr_B, float target_B,
                                uint32_t ch_C, float *curr_C, float target_C,
                                uint32_t speed_ms)
{
  float start_A, start_B, start_C;
  float delta_A, delta_B, delta_C, max_delta;
  uint32_t steps, i;

  target_A = Clamp_Angle(target_A);
  target_B = Clamp_Angle(target_B);
  target_C = Clamp_Angle(target_C);
  start_A = *curr_A;
  start_B = *curr_B;
  start_C = *curr_C;
  delta_A = target_A - start_A;
  delta_B = target_B - start_B;
  delta_C = target_C - start_C;
  max_delta = Abs_Float(delta_A);
  if (Abs_Float(delta_B) > max_delta) max_delta = Abs_Float(delta_B);
  if (Abs_Float(delta_C) > max_delta) max_delta = Abs_Float(delta_C);
  steps = Steps_For_Delta(max_delta);

  for (i = 1U; i <= steps; i++)
  {
    Service_Console();
    if (command_override != COMMAND_NONE) return;
    float t = (float)i / (float)steps;
    float progress = Smooth_Progress(t);
    *curr_A = start_A + delta_A * progress;
    *curr_B = start_B + delta_B * progress;
    *curr_C = start_C + delta_C * progress;

    __HAL_TIM_SET_COMPARE(&htim2, ch_A, Degree_To_Channel_Pulse(ch_A, *curr_A));
    __HAL_TIM_SET_COMPARE(&htim2, ch_B, Degree_To_Channel_Pulse(ch_B, *curr_B));
    __HAL_TIM_SET_COMPARE(&htim2, ch_C, Degree_To_Channel_Pulse(ch_C, *curr_C));
    Controlled_Delay(speed_ms);
  }
  *curr_A = target_A;
  *curr_B = target_B;
  *curr_C = target_C;
  __HAL_TIM_SET_COMPARE(&htim2, ch_A, Degree_To_Channel_Pulse(ch_A, target_A));
  __HAL_TIM_SET_COMPARE(&htim2, ch_B, Degree_To_Channel_Pulse(ch_B, target_B));
  __HAL_TIM_SET_COMPARE(&htim2, ch_C, Degree_To_Channel_Pulse(ch_C, target_C));
}

static void Set_Right_Compare(void)
{
  __HAL_TIM_SET_COMPARE(&htim16, TIM_CHANNEL_1, Degree_To_Pulse(right_ch1));
  __HAL_TIM_SET_COMPARE(&htim17, TIM_CHANNEL_1, Degree_To_Pulse(right_ch2));
  __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1, Degree_To_Pulse(right_ch3));
  __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_2, Degree_To_Pulse(right_ch4));
}

static void Move_Both_Arms_Smooth(float l1, float l2, float l3, float l4,
                                  float r1, float r2, float r3, float r4,
                                  uint32_t speed_ms)
{
  float *curr[8] = {
    &current_ch1, &current_ch2, &current_ch3, &current_ch4,
    &right_ch1, &right_ch2, &right_ch3, &right_ch4
  };
  float target[8] = {
    Clamp_Angle(l1), Clamp_Angle(l2), Clamp_Angle(l3), Clamp_Angle(l4),
    Clamp_Angle(r1), Clamp_Angle(r2), Clamp_Angle(r3), Clamp_Angle(r4)
  };
  float start[8];
  float delta[8];
  float max_delta = 0.0f;
  uint32_t steps;
  uint32_t i;
  uint32_t j;

  for (j = 0U; j < 8U; j++)
  {
    start[j] = *curr[j];
    delta[j] = target[j] - start[j];
    if (Abs_Float(delta[j]) > max_delta) max_delta = Abs_Float(delta[j]);
  }
  steps = Steps_For_Delta(max_delta);
  for (i = 1U; i <= steps; i++)
  {
    float progress;
    Service_Console();
    if (command_override != COMMAND_NONE) return;
    progress = Smooth_Progress((float)i / (float)steps);
    for (j = 0U; j < 8U; j++) *curr[j] = start[j] + delta[j] * progress;
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, Degree_To_Pulse(current_ch1));
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, Degree_To_Pulse(current_ch2));
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, Degree_To_Pulse(current_ch3));
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_4, Degree_To_Pulse(current_ch4));
    Set_Right_Compare();
    Controlled_Delay(speed_ms);
  }
  for (j = 0U; j < 8U; j++) *curr[j] = target[j];
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, Degree_To_Pulse(current_ch1));
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, Degree_To_Pulse(current_ch2));
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, Degree_To_Pulse(current_ch3));
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_4, Degree_To_Pulse(current_ch4));
  Set_Right_Compare();
}

void Move_Both_To_Loading_Safely(void)
{
  /* 먼저 상하 관절을 접고, 그 다음 J1만 회전한다. */
  Move_Both_Arms_Smooth(current_ch1, 120.0f, 56.0f, 0.0f,
                        right_ch1, 120.0f, 56.0f, 0.0f,
                        SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) return;
  Move_Both_Arms_Smooth(90.0f, 120.0f, 56.0f, 0.0f,
                        90.0f, 120.0f, 56.0f, 0.0f,
                        SERVO_MOVE_DELAY_MS);
}

void Move_Both_To_Home_Safely(void)
{
  /* 임의 자세에서도 회전과 상하 동작이 겹치지 않도록 로딩을 경유한다. */
  Move_Both_To_Loading_Safely();
  if (command_override != COMMAND_NONE) return;
  Move_Both_Arms_Smooth(0.0f, 120.0f, 56.0f, 0.0f,
                        0.0f, 120.0f, 56.0f, 0.0f,
                        SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) return;
  Move_Both_Arms_Smooth(0.0f, 90.0f, 90.0f, 90.0f,
                        0.0f, 90.0f, 90.0f, 90.0f,
                        SERVO_MOVE_DELAY_MS);
}

void Run_Pipeline_Camera_Plans(void)
{
  CameraPickPlan *pick = NULL;
  uint8_t pick_arm = 0U;
  float carrier_place = 90.0f;

  if (dual_left_present != 0U) { pick = &dual_left_plan; pick_arm = 1U; }
  if (dual_right_present != 0U) { pick = &dual_right_plan; pick_arm = 2U; }
  motion_busy = 1U;

  /* 첫 물체: 반대편 팔은 로딩에서 정지하고 선택된 팔만 집어서 전체 로딩으로 온다. */
  if (pipeline_holding_arm == 0U)
  {
    Left_Suction_Set(0U); Right_Suction_Set(0U);
    Move_Both_To_Loading_Safely();
    if (command_override != COMMAND_NONE) goto aborted;
    if (pick_arm == 1U)
    {
      Move_Both_Arms_Smooth(pick->j1,120,56,0, 90,120,56,0, SERVO_MOVE_DELAY_MS);
      Move_Both_Arms_Smooth(pick->j1,pick->approach[0],pick->approach[1],pick->approach[2], 90,120,56,0, SERVO_MOVE_DELAY_MS);
      Move_Both_Arms_Smooth(pick->j1,pick->contact[0],pick->contact[1],pick->contact[2], 90,120,56,0, SERVO_MOVE_DELAY_MS);
      Move_Both_Arms_Smooth(pick->j1,pick->preload[0],pick->preload[1],pick->preload[2], 90,120,56,0, SERVO_MOVE_DELAY_MS);
      Controlled_Delay(500U);
      if (command_override != COMMAND_NONE) goto aborted;
      Left_Suction_Set(1U);
      Controlled_Delay(500U);
      if (command_override != COMMAND_NONE) goto aborted;
      Move_Both_Arms_Smooth(pick->j1,pick->lift[0],pick->lift[1],pick->lift[2], 90,120,56,0, SERVO_MOVE_DELAY_MS);
      Move_Both_Arms_Smooth(pick->j1,120,56,0, 90,120,56,0, SERVO_MOVE_DELAY_MS);
      /* The picked object is lifted and the working joints are retracted.
         Camera preparation may overlap only the remaining J1 return. */
      Console_Write("CAMERA_CLEAR\r\n");
      Move_Both_Arms_Smooth(90,120,56,0, 90,120,56,0, SERVO_MOVE_DELAY_MS);
    }
    else
    {
      Move_Both_Arms_Smooth(90,120,56,0, pick->j1,120,56,0, SERVO_MOVE_DELAY_MS);
      Move_Both_Arms_Smooth(90,120,56,0, pick->j1,pick->approach[0],pick->approach[1],pick->approach[2], SERVO_MOVE_DELAY_MS);
      Move_Both_Arms_Smooth(90,120,56,0, pick->j1,pick->contact[0],pick->contact[1],pick->contact[2], SERVO_MOVE_DELAY_MS);
      Move_Both_Arms_Smooth(90,120,56,0, pick->j1,pick->preload[0],pick->preload[1],pick->preload[2], SERVO_MOVE_DELAY_MS);
      Controlled_Delay(500U);
      if (command_override != COMMAND_NONE) goto aborted;
      Right_Suction_Set(1U);
      Controlled_Delay(500U);
      if (command_override != COMMAND_NONE) goto aborted;
      Move_Both_Arms_Smooth(90,120,56,0, pick->j1,pick->lift[0],pick->lift[1],pick->lift[2], SERVO_MOVE_DELAY_MS);
      Move_Both_Arms_Smooth(90,120,56,0, pick->j1,120,56,0, SERVO_MOVE_DELAY_MS);
      /* The picked object is lifted and the working joints are retracted.
         Camera preparation may overlap only the remaining J1 return. */
      Console_Write("CAMERA_CLEAR\r\n");
      Move_Both_Arms_Smooth(90,120,56,0, 90,120,56,0, SERVO_MOVE_DELAY_MS);
    }
    if (command_override != COMMAND_NONE) goto aborted;
    pipeline_holding_arm = pick_arm;
    pipeline_holding_size_mm = pick->size_mm;
    Console_Write("HOLDING\r\n");
    motion_busy = 0U;
    return;
  }

  carrier_place = (pipeline_holding_arm == 1U)
      ? ((pipeline_holding_size_mm < 40.0f) ? 0.0f : 180.0f)
      : ((pipeline_holding_size_mm < 40.0f) ? 180.0f : 0.0f);

  /* 반대쪽에 다음 물체가 없으면 현재 든 물체만 안전하게 놓고 양팔 로딩 복귀. */
  if (pick_arm == 0U)
  {
    /* Both arms start this command at loading, clear of the work area. */
    Console_Write("CAMERA_CLEAR\r\n");
    if (pipeline_holding_arm == 1U)
    {
      Move_Both_Arms_Smooth(carrier_place,120,56,0, 90,120,56,0, SERVO_MOVE_DELAY_MS);
      Move_Both_Arms_Smooth(carrier_place,60,30,60, 90,120,56,0, SERVO_MOVE_DELAY_MS);
      Controlled_Delay(500U);
      if (command_override != COMMAND_NONE) goto aborted;
      Left_Suction_Set(0U);
      Controlled_Delay(1000U);
      if (command_override != COMMAND_NONE) goto aborted;
      Move_Both_Arms_Smooth(carrier_place,120,56,0, 90,120,56,0, SERVO_MOVE_DELAY_MS);
    }
    else
    {
      Move_Both_Arms_Smooth(90,120,56,0, carrier_place,120,56,0, SERVO_MOVE_DELAY_MS);
      Move_Both_Arms_Smooth(90,120,56,0, carrier_place,60,30,60, SERVO_MOVE_DELAY_MS);
      Controlled_Delay(500U);
      if (command_override != COMMAND_NONE) goto aborted;
      Right_Suction_Set(0U);
      Controlled_Delay(1000U);
      if (command_override != COMMAND_NONE) goto aborted;
      Move_Both_Arms_Smooth(90,120,56,0, carrier_place,120,56,0, SERVO_MOVE_DELAY_MS);
    }
    Move_Both_Arms_Smooth(90,120,56,0, 90,120,56,0, SERVO_MOVE_DELAY_MS);
    pipeline_holding_arm = 0U; pipeline_holding_size_mm = 0.0f;
    Console_Write("PLACED\r\n"); motion_busy = 0U; return;
  }

  /* 든 팔은 분류 상자로, 반대편 빈 팔은 다음 물체로 동시에 출발한다. */
  if (pipeline_holding_arm == 1U)
  {
    Move_Both_Arms_Smooth(carrier_place,120,56,0, pick->j1,120,56,0, SERVO_MOVE_DELAY_MS);
    Move_Both_Arms_Smooth(carrier_place,60,30,60, pick->j1,pick->approach[0],pick->approach[1],pick->approach[2], SERVO_MOVE_DELAY_MS);
    Move_Both_Arms_Smooth(carrier_place,60,30,60, pick->j1,pick->contact[0],pick->contact[1],pick->contact[2], SERVO_MOVE_DELAY_MS);
    Move_Both_Arms_Smooth(carrier_place,60,30,60, pick->j1,pick->preload[0],pick->preload[1],pick->preload[2], SERVO_MOVE_DELAY_MS);
    Controlled_Delay(500U);
    if (command_override != COMMAND_NONE) goto aborted;
    Left_Suction_Set(0U);
    Right_Suction_Set(1U);
    Controlled_Delay(1000U);
    if (command_override != COMMAND_NONE) goto aborted;
    Move_Both_Arms_Smooth(carrier_place,120,56,0, pick->j1,pick->lift[0],pick->lift[1],pick->lift[2], SERVO_MOVE_DELAY_MS);
    Move_Both_Arms_Smooth(carrier_place,120,56,0, pick->j1,120,56,0, SERVO_MOVE_DELAY_MS);
    /* Both working joints are retracted; overlap only the final J1 return. */
    Console_Write("CAMERA_CLEAR\r\n");
    Move_Both_Arms_Smooth(90,120,56,0, 90,120,56,0, SERVO_MOVE_DELAY_MS);
  }
  else
  {
    Move_Both_Arms_Smooth(pick->j1,120,56,0, carrier_place,120,56,0, SERVO_MOVE_DELAY_MS);
    Move_Both_Arms_Smooth(pick->j1,pick->approach[0],pick->approach[1],pick->approach[2], carrier_place,60,30,60, SERVO_MOVE_DELAY_MS);
    Move_Both_Arms_Smooth(pick->j1,pick->contact[0],pick->contact[1],pick->contact[2], carrier_place,60,30,60, SERVO_MOVE_DELAY_MS);
    Move_Both_Arms_Smooth(pick->j1,pick->preload[0],pick->preload[1],pick->preload[2], carrier_place,60,30,60, SERVO_MOVE_DELAY_MS);
    Controlled_Delay(500U);
    if (command_override != COMMAND_NONE) goto aborted;
    Right_Suction_Set(0U);
    Left_Suction_Set(1U);
    Controlled_Delay(1000U);
    if (command_override != COMMAND_NONE) goto aborted;
    Move_Both_Arms_Smooth(pick->j1,pick->lift[0],pick->lift[1],pick->lift[2], carrier_place,120,56,0, SERVO_MOVE_DELAY_MS);
    Move_Both_Arms_Smooth(pick->j1,120,56,0, carrier_place,120,56,0, SERVO_MOVE_DELAY_MS);
    /* Both working joints are retracted; overlap only the final J1 return. */
    Console_Write("CAMERA_CLEAR\r\n");
    Move_Both_Arms_Smooth(90,120,56,0, 90,120,56,0, SERVO_MOVE_DELAY_MS);
  }
  if (command_override != COMMAND_NONE) goto aborted;
  pipeline_holding_arm = pick_arm;
  pipeline_holding_size_mm = pick->size_mm;
  Console_Write("CROSS_DONE\r\n"); motion_busy = 0U; return;

aborted:
  Left_Suction_Set(0U); Right_Suction_Set(0U);
  pipeline_holding_arm = 0U; pipeline_holding_size_mm = 0.0f;
  Console_Write("ERROR\r\n"); motion_busy = 0U;
}

void Run_Dual_Camera_Plans(void)
{
  CameraPickPlan left = dual_left_plan;
  CameraPickPlan right = dual_right_plan;
  float left_place = (left.size_mm < 40.0f) ? 0.0f : 180.0f;
  float right_place = (right.size_mm < 40.0f) ? 180.0f : 0.0f;
  float idle_l1 = ((dual_suction_enabled == 0U) && (dual_left_present == 0U)) ? current_ch1 : 90.0f;
  float idle_l2 = ((dual_suction_enabled == 0U) && (dual_left_present == 0U)) ? current_ch2 : 120.0f;
  float idle_l3 = ((dual_suction_enabled == 0U) && (dual_left_present == 0U)) ? current_ch3 : 56.0f;
  float idle_l4 = ((dual_suction_enabled == 0U) && (dual_left_present == 0U)) ? current_ch4 : 0.0f;

  motion_busy = 1U;
  Left_Suction_Set(0U);
  Right_Suction_Set(0U);

  /* 상하 관절을 먼저 로딩 형태로 만든 뒤 J1만 회전한다. */
  Move_Both_Arms_Smooth(current_ch1, idle_l2, idle_l3, idle_l4,
                        right_ch1, 120.0f, 56.0f, 0.0f,
                        SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;
  Move_Both_Arms_Smooth(idle_l1, idle_l2, idle_l3, idle_l4,
                        90.0f, 120.0f, 56.0f, 0.0f,
                        SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;

  Move_Both_Arms_Smooth(
      (dual_left_present != 0U) ? left.j1 : idle_l1, idle_l2, idle_l3, idle_l4,
      (dual_right_present != 0U) ? right.j1 : 90.0f, 120.0f, 56.0f, 0.0f,
      EMPTY_ARM_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;

#define LP(P, I, D) ((dual_left_present != 0U) ? (P)[I] : (D))
#define RP(P, I, D) ((dual_right_present != 0U) ? (P)[I] : (D))
  Move_Both_Arms_Smooth(
      (dual_left_present != 0U) ? left.j1 : idle_l1,
      LP(left.approach, 0, idle_l2), LP(left.approach, 1, idle_l3), LP(left.approach, 2, idle_l4),
      (dual_right_present != 0U) ? right.j1 : 90.0f,
      RP(right.approach, 0, 120.0f), RP(right.approach, 1, 56.0f), RP(right.approach, 2, 0.0f),
      SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;
  Move_Both_Arms_Smooth(
      (dual_left_present != 0U) ? left.j1 : idle_l1,
      LP(left.contact, 0, idle_l2), LP(left.contact, 1, idle_l3), LP(left.contact, 2, idle_l4),
      (dual_right_present != 0U) ? right.j1 : 90.0f,
      RP(right.contact, 0, 120.0f), RP(right.contact, 1, 56.0f), RP(right.contact, 2, 0.0f),
      SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;
  Move_Both_Arms_Smooth(
      (dual_left_present != 0U) ? left.j1 : idle_l1,
      LP(left.preload, 0, idle_l2), LP(left.preload, 1, idle_l3), LP(left.preload, 2, idle_l4),
      (dual_right_present != 0U) ? right.j1 : 90.0f,
      RP(right.preload, 0, 120.0f), RP(right.preload, 1, 56.0f), RP(right.preload, 2, 0.0f),
      SERVO_MOVE_DELAY_MS);
  Controlled_Delay(500U);
  if (command_override != COMMAND_NONE) goto aborted;

  if ((dual_suction_enabled != 0U) && (dual_left_present != 0U)) Left_Suction_Set(1U);
  if ((dual_suction_enabled != 0U) && (dual_right_present != 0U)) Right_Suction_Set(1U);
  Controlled_Delay(500U);
  if (command_override != COMMAND_NONE) goto aborted;

  Move_Both_Arms_Smooth(
      (dual_left_present != 0U) ? left.j1 : idle_l1,
      LP(left.lift, 0, idle_l2), LP(left.lift, 1, idle_l3), LP(left.lift, 2, idle_l4),
      (dual_right_present != 0U) ? right.j1 : 90.0f,
      RP(right.lift, 0, 120.0f), RP(right.lift, 1, 56.0f), RP(right.lift, 2, 0.0f),
      SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;

  /* 물체 상승 후 J2/J3/J4를 먼저 접고, 마지막에 J1만 로딩 복귀한다. */
  Move_Both_Arms_Smooth(
      (dual_left_present != 0U) ? left.j1 : idle_l1,
      idle_l2, idle_l3, idle_l4,
      (dual_right_present != 0U) ? right.j1 : 90.0f,
      120.0f, 56.0f, 0.0f,
      SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;
  Move_Both_Arms_Smooth(idle_l1, idle_l2, idle_l3, idle_l4,
                        90.0f, 120.0f, 56.0f, 0.0f,
                        SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;
  /* Both picked objects are now outside the camera work area. */
  Console_Write("CAMERA_CLEAR\r\n");
  Move_Both_Arms_Smooth(
      (dual_left_present != 0U) ? left_place : idle_l1, idle_l2, idle_l3, idle_l4,
      (dual_right_present != 0U) ? right_place : 90.0f, 120.0f, 56.0f, 0.0f,
      EMPTY_ARM_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;
  Move_Both_Arms_Smooth(
      (dual_left_present != 0U) ? left_place : idle_l1,
      (dual_left_present != 0U) ? 60.0f : idle_l2,
      (dual_left_present != 0U) ? 30.0f : idle_l3,
      (dual_left_present != 0U) ? 60.0f : idle_l4,
      (dual_right_present != 0U) ? right_place : 90.0f,
      (dual_right_present != 0U) ? 60.0f : 120.0f,
      (dual_right_present != 0U) ? 30.0f : 56.0f,
      (dual_right_present != 0U) ? 60.0f : 0.0f,
      SERVO_MOVE_DELAY_MS);
  Controlled_Delay(500U);
  if (command_override != COMMAND_NONE) goto aborted;

  if (dual_left_present != 0U) Left_Suction_Set(0U);
  if (dual_right_present != 0U) Right_Suction_Set(0U);
  Controlled_Delay(1000U);
  if (command_override != COMMAND_NONE) goto aborted;

  /*
   * 놓기 위치에서 기둥과 충돌하지 않도록 2단계로 복귀한다.
   * 1단계: J1은 놓기 방향에 고정하고 J2/J3/J4만 로딩 자세로 만든다.
   */
  Move_Both_Arms_Smooth(
      (dual_left_present != 0U) ? left_place : idle_l1,
      (dual_left_present != 0U) ? 120.0f : idle_l2,
      (dual_left_present != 0U) ? 56.0f : idle_l3,
      (dual_left_present != 0U) ? 0.0f : idle_l4,
      (dual_right_present != 0U) ? right_place : 90.0f,
      120.0f, 56.0f, 0.0f,
      EMPTY_ARM_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;

  /* 2단계: 팔을 접은 뒤 마지막으로 J1만 로딩 각도 90도로 복귀한다. */
  Move_Both_Arms_Smooth(idle_l1, idle_l2, idle_l3, idle_l4,
                        90.0f, 120.0f, 56.0f, 0.0f,
                        EMPTY_ARM_MOVE_DELAY_MS);
  Controlled_Delay(200U);
  if (command_override != COMMAND_NONE) goto aborted;

  Console_Write("DONE\r\n");
  motion_busy = 0U;
#undef LP
#undef RP
  return;

aborted:
  Left_Suction_Set(0U);
  Right_Suction_Set(0U);
  Console_Write("ERROR\r\n");
  motion_busy = 0U;
#undef LP
#undef RP
}

void Go_Home(void)
{
  motion_busy = 1U;
  Console_Write("[기본 자세] 충돌 방지를 위해 J1을 먼저 왼쪽 0도로 이동합니다.\r\n");
  Move_Single_Channel_Smooth(TIM_CHANNEL_1, &current_ch1, 0.0f,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[기본 자세] J4를 두 번째로 90도로 이동합니다.\r\n");
  Move_Single_Channel_Smooth(TIM_CHANNEL_4, &current_ch4, 90.0f,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[기본 자세] J2와 J3를 마지막으로 함께 90도로 이동합니다.\r\n");
  Move_Dual_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 90.0f,
                           TIM_CHANNEL_3, &current_ch3, 90.0f,
                           SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto done;
  if (command_override == COMMAND_NONE)
    Console_Write("[완료] 안전 기본 자세 0/90/90/90입니다.\r\n");
done:
  motion_busy = 0U;
}

void Go_Loading(void)
{
  motion_busy = 1U;
  Console_Write("[로딩 자세] J2/J3를 120도/56도로 이동합니다.\r\n");
  Move_Dual_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 120.0f,
                           TIM_CHANNEL_3, &current_ch3, 56.0f,
                           SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[로딩 자세] J4를 0도로 이동합니다.\r\n");
  Move_Single_Channel_Smooth(TIM_CHANNEL_4, &current_ch4, 0.0f,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[로딩 자세] J1을 마지막으로 정면 90도로 이동합니다.\r\n");
  Move_Single_Channel_Smooth(TIM_CHANNEL_1, &current_ch1, 90.0f,
                             SERVO_MOVE_DELAY_MS);
  if (command_override == COMMAND_NONE)
    Console_Write("[완료] 로딩 자세 정렬을 완료했습니다.\r\n");
done:
  motion_busy = 0U;
}

void Run_Camera_Plan(void)
{
  CameraPickPlan plan = camera_plan;
  float place_j1 = (plan.size_mm < 40.0f) ? 0.0f : 180.0f;

  motion_busy = 1U;
  Console_Write("[카메라] 동적 피킹 명령 실행을 시작합니다.\r\n");
  Go_Loading();
  motion_busy = 1U;
  if (command_override != COMMAND_NONE) goto aborted;

  Console_Write("[카메라 1/15] J1을 물체 방향으로 먼저 이동합니다.\r\n");
  Move_Single_Channel_Smooth(TIM_CHANNEL_1, &current_ch1, plan.j1,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;

  Console_Write("[카메라 2/15] 물체 상부로 접근합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, plan.approach[0],
                             TIM_CHANNEL_3, &current_ch3, plan.approach[1],
                             TIM_CHANNEL_4, &current_ch4, plan.approach[2],
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;

  Console_Write("[카메라 3/15] 물체 상판에 접촉합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, plan.contact[0],
                             TIM_CHANNEL_3, &current_ch3, plan.contact[1],
                             TIM_CHANNEL_4, &current_ch4, plan.contact[2],
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;

  Console_Write("[카메라 4/15] 예압 자세로 이동합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, plan.preload[0],
                             TIM_CHANNEL_3, &current_ch3, plan.preload[1],
                             TIM_CHANNEL_4, &current_ch4, plan.preload[2],
                             SERVO_MOVE_DELAY_MS);
  Console_Write("[카메라 4/15] 예압 도착 후 0.5초 안정화합니다.\r\n");
  Controlled_Delay(500U);
  if (command_override != COMMAND_NONE) goto aborted;

  Left_Suction_Set(1U);
  Console_Write("[카메라 4/15] 왼쪽 흡착 CH4 ON 후 0.5초 대기합니다.\r\n");
  Controlled_Delay(500U);
  if (command_override != COMMAND_NONE) goto aborted;

  Console_Write("[카메라 5/15] 상승 자세로 이동합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, plan.lift[0],
                             TIM_CHANNEL_3, &current_ch3, plan.lift[1],
                             TIM_CHANNEL_4, &current_ch4, plan.lift[2],
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;

  Console_Write("[카메라 6/15] J2/J3/J4부터 로딩으로 복귀합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 120.0f,
                             TIM_CHANNEL_3, &current_ch3, 56.0f,
                             TIM_CHANNEL_4, &current_ch4, 0.0f,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;

  Console_Write("[카메라 7/15] J1을 마지막으로 90도 복귀합니다.\r\n");
  Move_Single_Channel_Smooth(TIM_CHANNEL_1, &current_ch1, 90.0f,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;

  Console_Write("[카메라 8/15] J1을 분류 위치로 먼저 이동합니다.\r\n");
  Move_Single_Channel_Smooth(TIM_CHANNEL_1, &current_ch1, place_j1,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;

  Console_Write("[카메라 9/15] J2/J3/J4를 놓기 자세로 이동합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 60.0f,
                             TIM_CHANNEL_3, &current_ch3, 30.0f,
                             TIM_CHANNEL_4, &current_ch4, 60.0f,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;

  Console_Write("[카메라 10/15] 놓기 자세 도착 후 공압을 유지하며 0.5초 안정화합니다.\r\n");
  Controlled_Delay(500U);
  if (command_override != COMMAND_NONE) goto aborted;

  Left_Suction_Set(0U);
  Console_Write("[카메라 11/15] 왼쪽 흡착 CH4 OFF 후 1초 대기합니다.\r\n");
  Controlled_Delay(1000U);
  if (command_override != COMMAND_NONE) goto aborted;

  Console_Write("[카메라 12/15] J2/J3/J4부터 로딩으로 복귀합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 120.0f,
                             TIM_CHANNEL_3, &current_ch3, 56.0f,
                             TIM_CHANNEL_4, &current_ch4, 0.0f,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto aborted;

  Console_Write("[카메라 13/15] J1을 마지막으로 90도 복귀합니다.\r\n");
  Move_Single_Channel_Smooth(TIM_CHANNEL_1, &current_ch1, 90.0f,
                             SERVO_MOVE_DELAY_MS);
  Controlled_Delay(2000U);
  if (command_override != COMMAND_NONE) goto aborted;

  Console_Write("[카메라 14/15] 물체 처리 완료 · 로딩 자세에서 다음 명령을 기다립니다.\r\n");
  Console_Write("[카메라 15/15] 카메라가 빈 작업영역을 확인하면 안전 복귀합니다.\r\n");
  Console_Write("DONE\r\n");
  motion_busy = 0U;
  return;

aborted:
  Left_Suction_Set(0U);
  Console_Write("ERROR\r\n");
  motion_busy = 0U;
}

void Run_Pick_Cycle(void)
{
  motion_busy = 1U;
  Console_Write("[동작 1/16] 로딩 자세로 이동합니다.\r\n");
  Go_Loading();
  motion_busy = 1U;
  if (command_override != COMMAND_NONE) goto done;

  /* Corrected J1 convention: 90=front, 0=robot-left, 180=robot-right. */
  Console_Write("[동작 2/16] J1을 큐브 방향 63.5도로 먼저 회전합니다.\r\n");
  Move_Single_Channel_Smooth(TIM_CHANNEL_1, &current_ch1, 63.5f,
                             SERVO_MOVE_DELAY_MS);
  Controlled_Delay(1000U);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[동작 3/16] 큐브 상부로 접근합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 62.0f,
                             TIM_CHANNEL_3, &current_ch3, 24.5f,
                             TIM_CHANNEL_4, &current_ch4, 22.5f,
                             SERVO_MOVE_DELAY_MS);
  Controlled_Delay(1500U);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[동작 4/16] 큐브 상판까지 내려가 접촉합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 60.5f,
                             TIM_CHANNEL_3, &current_ch3, 27.0f,
                             TIM_CHANNEL_4, &current_ch4, 26.5f,
                             SERVO_MOVE_DELAY_MS);
  Controlled_Delay(1500U);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[동작 5/16] 조인트를 움직여 2 mm 예압합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 60.0f,
                             TIM_CHANNEL_3, &current_ch3, 27.5f,
                             TIM_CHANNEL_4, &current_ch4, 27.5f,
                             SERVO_MOVE_DELAY_MS);
  Console_Write("[동작 5/16] 예압 도착 후 0.5초 안정화합니다.\r\n");
  Controlled_Delay(500U);
  if (command_override != COMMAND_NONE) goto done;

  Left_Suction_Set(1U);
  Console_Write("[동작 5/16] 왼쪽 흡착 CH4 ON 후 0.5초 대기합니다.\r\n");
  Controlled_Delay(500U);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[동작 6/16] 큐브를 총 16 mm 들어 올립니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 62.0f,
                             TIM_CHANNEL_3, &current_ch3, 25.5f,
                             TIM_CHANNEL_4, &current_ch4, 23.5f,
                             SERVO_MOVE_DELAY_MS);
  Controlled_Delay(1500U);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[동작 7/16] J1을 유지하고 J2/J3/J4부터 로딩으로 복귀합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 120.0f,
                             TIM_CHANNEL_3, &current_ch3, 56.0f,
                             TIM_CHANNEL_4, &current_ch4, 0.0f,
                             SERVO_MOVE_DELAY_MS);
  Controlled_Delay(1000U);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[동작 8/16] J1을 마지막으로 90도 복귀합니다.\r\n");
  Move_Single_Channel_Smooth(TIM_CHANNEL_1, &current_ch1, 90.0f,
                             SERVO_MOVE_DELAY_MS);

  Console_Write("[동작 9/16] 로딩 자세 도착 후 분류 위치로 이동합니다.\r\n");
  if (command_override != COMMAND_NONE) goto done;

  /* The current camera-free test cube is 30 mm, so sort it to robot-left. */
  Console_Write("[동작 10/16] J1을 먼저 왼쪽 0도로 회전합니다.\r\n");
  Move_Single_Channel_Smooth(TIM_CHANNEL_1, &current_ch1, 0.0f,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[동작 11/16] J2/J3/J4를 놓기 자세 60/30/60도로 이동합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 60.0f,
                             TIM_CHANNEL_3, &current_ch3, 30.0f,
                             TIM_CHANNEL_4, &current_ch4, 60.0f,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[동작 12/16] 놓기 자세에서 공압을 유지하며 0.5초 안정화합니다.\r\n");
  Controlled_Delay(500U);
  if (command_override != COMMAND_NONE) goto done;

  Left_Suction_Set(0U);
  Console_Write("[동작 13/16] 왼쪽 흡착 CH4 OFF로 큐브를 놓습니다.\r\n");
  Console_Write("[동작 14/16] 큐브를 놓은 뒤 1초 대기합니다.\r\n");
  Controlled_Delay(1000U);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[동작 15/16] J1을 유지하고 J2/J3/J4부터 로딩으로 복귀합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 120.0f,
                             TIM_CHANNEL_3, &current_ch3, 56.0f,
                             TIM_CHANNEL_4, &current_ch4, 0.0f,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[동작 16/16] J1을 마지막으로 90도 복귀합니다.\r\n");
  Move_Single_Channel_Smooth(TIM_CHANNEL_1, &current_ch1, 90.0f,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[완료] 분류 완료 후 로딩 자세에서 2초 대기합니다.\r\n");
  Controlled_Delay(2000U);
  if (command_override != COMMAND_NONE) goto done;

  /* Second object: 50 mm cube at Camera X=-71, Y=74, Z=50 mm. */
  Console_Write("\r\n[5 cm 큐브] 두 번째 피킹 및 오른쪽 분류를 시작합니다.\r\n");

  Console_Write("[5 cm 1/15] J1을 큐브 방향 122도로 먼저 회전합니다.\r\n");
  Move_Single_Channel_Smooth(TIM_CHANNEL_1, &current_ch1, 122.0f,
                             SERVO_MOVE_DELAY_MS);
  Controlled_Delay(1000U);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[5 cm 2/15] 큐브 상부 10 mm 위치로 접근합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 77.0f,
                             TIM_CHANNEL_3, &current_ch3, 36.0f,
                             TIM_CHANNEL_4, &current_ch4, 19.0f,
                             SERVO_MOVE_DELAY_MS);
  Controlled_Delay(1500U);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[5 cm 3/15] 큐브 상판까지 내려가 접촉합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 75.5f,
                             TIM_CHANNEL_3, &current_ch3, 38.5f,
                             TIM_CHANNEL_4, &current_ch4, 23.0f,
                             SERVO_MOVE_DELAY_MS);
  Controlled_Delay(1500U);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[5 cm 4/15] 조인트를 움직여 2 mm 예압합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 75.0f,
                             TIM_CHANNEL_3, &current_ch3, 39.0f,
                             TIM_CHANNEL_4, &current_ch4, 24.0f,
                             SERVO_MOVE_DELAY_MS);
  Console_Write("[5 cm 4/15] 예압 도착 후 0.5초 안정화합니다.\r\n");
  Controlled_Delay(500U);
  if (command_override != COMMAND_NONE) goto done;

  Left_Suction_Set(1U);
  Console_Write("[5 cm 4/15] 왼쪽 흡착 CH4 ON 후 0.5초 대기합니다.\r\n");
  Controlled_Delay(500U);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[5 cm 5/15] 큐브를 총 16 mm 들어 올립니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 76.5f,
                             TIM_CHANNEL_3, &current_ch3, 36.0f,
                             TIM_CHANNEL_4, &current_ch4, 19.5f,
                             SERVO_MOVE_DELAY_MS);
  Controlled_Delay(1500U);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[5 cm 6/15] J1을 유지하고 J2/J3/J4부터 로딩으로 복귀합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 120.0f,
                             TIM_CHANNEL_3, &current_ch3, 56.0f,
                             TIM_CHANNEL_4, &current_ch4, 0.0f,
                             SERVO_MOVE_DELAY_MS);
  Controlled_Delay(1000U);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[5 cm 7/15] J1을 마지막으로 90도 복귀합니다.\r\n");
  Move_Single_Channel_Smooth(TIM_CHANNEL_1, &current_ch1, 90.0f,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[5 cm 8/15] 로딩 자세 도착 후 분류 위치로 이동합니다.\r\n");
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[5 cm 9/15] J1을 먼저 오른쪽 180도로 회전합니다.\r\n");
  Move_Single_Channel_Smooth(TIM_CHANNEL_1, &current_ch1, 180.0f,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[5 cm 10/15] J2/J3/J4를 놓기 자세 60/30/60도로 이동합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 60.0f,
                             TIM_CHANNEL_3, &current_ch3, 30.0f,
                             TIM_CHANNEL_4, &current_ch4, 60.0f,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[5 cm 11/15] 놓기 자세에서 공압을 유지하며 0.5초 안정화합니다.\r\n");
  Controlled_Delay(500U);
  if (command_override != COMMAND_NONE) goto done;

  Left_Suction_Set(0U);
  Console_Write("[5 cm 12/15] 왼쪽 흡착 CH4 OFF로 큐브를 놓습니다.\r\n");
  Console_Write("[5 cm 13/15] 큐브를 놓은 뒤 1초 대기합니다.\r\n");
  Controlled_Delay(1000U);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[5 cm 14/15] J1을 유지하고 J2/J3/J4부터 로딩으로 복귀합니다.\r\n");
  Move_Triple_Channel_Smooth(TIM_CHANNEL_2, &current_ch2, 120.0f,
                             TIM_CHANNEL_3, &current_ch3, 56.0f,
                             TIM_CHANNEL_4, &current_ch4, 0.0f,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[5 cm 15/15] J1을 마지막으로 90도 복귀합니다.\r\n");
  Move_Single_Channel_Smooth(TIM_CHANNEL_1, &current_ch1, 90.0f,
                             SERVO_MOVE_DELAY_MS);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[완료] 3 cm와 5 cm 큐브 분류를 모두 완료했습니다.\r\n");
  Console_Write("[대기] 로딩 자세에서 2초 대기합니다.\r\n");
  Controlled_Delay(2000U);
  if (command_override != COMMAND_NONE) goto done;

  Console_Write("[복귀] 안전한 전원 종료를 위해 0/90/90/90 자세로 복귀합니다.\r\n");
  Go_Home();
done:
  Left_Suction_Set(0U);
  motion_busy = 0U;
}
/* USER CODE END 0 */

int main(void)
{
  HAL_Init();
  SystemClock_Config();
  PeriphCommonClock_Config();
  MX_GPIO_Init();
  MX_TIM1_Init();
  MX_TIM2_Init();
  MX_TIM16_Init();
  MX_TIM17_Init();

  /* USER CODE BEGIN 2 */
  Left_Suction_Set(0U);
  Right_Suction_Set(0U);

  current_ch1 = 0.0f;
  current_ch2 = 90.0f;
  current_ch3 = 90.0f;
  current_ch4 = 90.0f;
  right_ch1 = 0.0f;
  right_ch2 = 90.0f;
  right_ch3 = 90.0f;
  right_ch4 = 90.0f;

  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, Degree_To_Channel_Pulse(TIM_CHANNEL_1, current_ch1));
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, Degree_To_Channel_Pulse(TIM_CHANNEL_2, current_ch2));
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, Degree_To_Channel_Pulse(TIM_CHANNEL_3, current_ch3));
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_4, Degree_To_Channel_Pulse(TIM_CHANNEL_4, current_ch4));
  Set_Right_Compare();

  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_3);
  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_4);
  HAL_TIM_PWM_Start(&htim16, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim17, TIM_CHANNEL_1);

  HAL_Delay(500);

  BSP_LED_Init(LED_BLUE);
  BSP_LED_Init(LED_GREEN);
  BSP_LED_Init(LED_RED);

  BSP_PB_Init(BUTTON_SW1, BUTTON_MODE_EXTI);
  BSP_PB_Init(BUTTON_SW2, BUTTON_MODE_EXTI);
  BSP_PB_Init(BUTTON_SW3, BUTTON_MODE_EXTI);

  BspCOMInit.BaudRate   = 115200;
  BspCOMInit.WordLength = COM_WORDLENGTH_8B;
  BspCOMInit.StopBits   = COM_STOPBITS_1;
  BspCOMInit.Parity     = COM_PARITY_NONE;
  BspCOMInit.HwFlowCtl  = COM_HWCONTROL_NONE;
  if (BSP_COM_Init(COM1, &BspCOMInit) != BSP_ERROR_NONE)
  {
    Error_Handler();
  }

  BSP_LED_On(LED_BLUE);

  Console_Print_Guide();
  BSP_LED_Off(LED_BLUE);
  BSP_LED_On(LED_GREEN);
  /* USER CODE END 2 */

  while (1)
  {
    Service_Console();

    if (command_override == COMMAND_STOP)
    {
      command_override = COMMAND_NONE;
      motion_busy = 0U;
      Left_Suction_Set(0U);
      Right_Suction_Set(0U);
      pipeline_holding_arm = 0U;
      pipeline_holding_size_mm = 0.0f;
      Console_Write("EMERGENCY_DONE\r\n");
    }
    else if (command_override == COMMAND_HOME)
    {
      command_override = COMMAND_NONE;
      motion_busy = 1U;
      pipeline_holding_arm = 0U;
      pipeline_holding_size_mm = 0.0f;
      Move_Both_To_Home_Safely();
      motion_busy = 0U;
      if (emergency_stop_requested != 0U)
      {
        emergency_stop_requested = 0U;
        Console_Write("EMERGENCY_DONE\r\n");
      }
    }
    else if (command_override == COMMAND_LOADING)
    {
      command_override = COMMAND_NONE;
      motion_busy = 1U;
      Move_Both_To_Loading_Safely();
      motion_busy = 0U;
    }
    else if (finish_requested != 0U)
    {
      finish_requested = 0U;
      pipeline_holding_arm = 0U;
      pipeline_holding_size_mm = 0.0f;
      Console_Write("[작업 종료] 물체 없음 확인 · 안전 기본 자세로 복귀합니다.\r\n");
      Move_Both_To_Home_Safely();
      Console_Write("FINISHED\r\n");
    }
    else if (dual_plan_ready != 0U)
    {
      dual_plan_ready = 0U;
      if (dual_pipeline_requested != 0U)
        Run_Pipeline_Camera_Plans();
      else
        Run_Dual_Camera_Plans();
    }
    else if (camera_plan_ready != 0U)
    {
      camera_plan_ready = 0U;
      Run_Camera_Plan();
    }
    else if (start_requested != 0U)
    {
      start_requested = 0U;
      Run_Pick_Cycle();
    }

    HAL_Delay(10U);
  }
}

void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI | RCC_OSCILLATORTYPE_MSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.MSIState = RCC_MSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.MSICalibrationValue = RCC_MSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_10;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_MSI;
  RCC_OscInitStruct.PLL.PLLM = RCC_PLLM_DIV4;
  RCC_OscInitStruct.PLL.PLLN = 32;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) Error_Handler();

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK4 | RCC_CLOCKTYPE_HCLK2 |
                                RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                                RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.AHBCLK2Divider = RCC_SYSCLK_DIV2;
  RCC_ClkInitStruct.AHBCLK4Divider = RCC_SYSCLK_DIV2;
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK) Error_Handler();
}

void PeriphCommonClock_Config(void)
{
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_SMPS;
  PeriphClkInitStruct.SmpsClockSelection = RCC_SMPSCLKSOURCE_HSI;
  PeriphClkInitStruct.SmpsDivSelection = RCC_SMPSCLKDIV_RANGE0;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK) Error_Handler();
}

static void MX_TIM1_Init(void)
{
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};
  TIM_BreakDeadTimeConfigTypeDef sBreakDeadTimeConfig = {0};

  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 63;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 19999;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim1) != HAL_OK) Error_Handler();

  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterOutputTrigger2 = TIM_TRGO2_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK) Error_Handler();

  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 1500;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCNPolarity = TIM_OCNPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  sConfigOC.OCIdleState = TIM_OCIDLESTATE_RESET;
  sConfigOC.OCNIdleState = TIM_OCNIDLESTATE_RESET;
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_1) != HAL_OK) Error_Handler();
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_2) != HAL_OK) Error_Handler();

  sBreakDeadTimeConfig.OffStateRunMode = TIM_OSSR_DISABLE;
  sBreakDeadTimeConfig.OffStateIDLEMode = TIM_OSSI_DISABLE;
  sBreakDeadTimeConfig.LockLevel = TIM_LOCKLEVEL_OFF;
  sBreakDeadTimeConfig.DeadTime = 0;
  sBreakDeadTimeConfig.BreakState = TIM_BREAK_DISABLE;
  sBreakDeadTimeConfig.BreakPolarity = TIM_BREAKPOLARITY_HIGH;
  sBreakDeadTimeConfig.BreakFilter = 0;
  sBreakDeadTimeConfig.BreakAFMode = TIM_BREAK_AFMODE_INPUT;
  sBreakDeadTimeConfig.Break2State = TIM_BREAK2_DISABLE;
  sBreakDeadTimeConfig.Break2Polarity = TIM_BREAK2POLARITY_HIGH;
  sBreakDeadTimeConfig.Break2Filter = 0;
  sBreakDeadTimeConfig.Break2AFMode = TIM_BREAK_AFMODE_INPUT;
  sBreakDeadTimeConfig.AutomaticOutput = TIM_AUTOMATICOUTPUT_DISABLE;
  if (HAL_TIMEx_ConfigBreakDeadTime(&htim1, &sBreakDeadTimeConfig) != HAL_OK) Error_Handler();
  HAL_TIM_MspPostInit(&htim1);
}

static void MX_TIM2_Init(void)
{
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 64 - 1;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 20000 - 1;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim2) != HAL_OK) Error_Handler();

  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK) Error_Handler();

  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 500;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;

  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK) Error_Handler();
  sConfigOC.Pulse = 1500;
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_2) != HAL_OK) Error_Handler();
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_3) != HAL_OK) Error_Handler();
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_4) != HAL_OK) Error_Handler();

  HAL_TIM_MspPostInit(&htim2);
}

static void MX_TIM16_Init(void)
{
  TIM_OC_InitTypeDef sConfigOC = {0};

  htim16.Instance = TIM16;
  htim16.Init.Prescaler = 63;
  htim16.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim16.Init.Period = 19999;
  htim16.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim16.Init.RepetitionCounter = 0;
  htim16.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim16) != HAL_OK) Error_Handler();
  if (HAL_TIM_PWM_Init(&htim16) != HAL_OK) Error_Handler();
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 500;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim16, &sConfigOC, TIM_CHANNEL_1) != HAL_OK) Error_Handler();
  HAL_TIM_MspPostInit(&htim16);
}

static void MX_TIM17_Init(void)
{
  TIM_OC_InitTypeDef sConfigOC = {0};

  htim17.Instance = TIM17;
  htim17.Init.Prescaler = 63;
  htim17.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim17.Init.Period = 19999;
  htim17.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim17.Init.RepetitionCounter = 0;
  htim17.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim17) != HAL_OK) Error_Handler();
  if (HAL_TIM_PWM_Init(&htim17) != HAL_OK) Error_Handler();
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 1500;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim17, &sConfigOC, TIM_CHANNEL_1) != HAL_OK) Error_Handler();
  HAL_TIM_MspPostInit(&htim17);
}

static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();

  /* Low Active pneumatic outputs remain OFF while GPIO modes are configured. */
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_11, GPIO_PIN_SET);
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_4, GPIO_PIN_SET);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_12 | GPIO_PIN_13 | GPIO_PIN_14 | GPIO_PIN_15,
                    GPIO_PIN_RESET);

  /* Pneumatic CH1 (right) and CH4 (left), external pull-up/open-drain drive. */
  GPIO_InitStruct.Pin = GPIO_PIN_11;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_OD;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = GPIO_PIN_4;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_OD;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_3;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF1_TIM2;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = GPIO_PIN_12 | GPIO_PIN_13 | GPIO_PIN_14 | GPIO_PIN_15;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = GPIO_PIN_11 | GPIO_PIN_12;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF10_USB;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}

void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
  }
}
