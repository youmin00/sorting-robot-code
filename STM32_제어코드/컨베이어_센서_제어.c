/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Conveyor Belt + RealSense Python Link Test
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <string.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/*
  핀 구성

  PA8  = TIM1_CH1 PWM 출력
       = 스텝모터 드라이버 STEP 핀

  PA9  = 방향 또는 Enable 출력 핀
       = 기존 코드에서 S 입력 시 HIGH로 설정하던 핀

  PA15 = Active-LOW 센서 입력
       = 물체 없음: HIGH
       = 물체 감지: LOW

  현재 테스트:
  - 센서는 아직 STM32에서 분리해둠
  - 하지만 PA15 센서 감지 코드는 그대로 유지
  - PA15는 Pull-up 설정으로 기본 HIGH 유지
*/
#define SENSOR_RESTART_IGNORE_MS 1000U
#define WALL_DETECT_DEBOUNCE_MS    30U

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef htim1;
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
uint8_t rx_data;

/*
  belt_running:
  0 = 벨트 정지
  1 = 벨트 이동 중

  auto_mode:
  0 = 자동 반복 꺼짐
  1 = 자동 반복 켜짐

  waiting_camera:
  0 = 카메라 응답 대기 아님
  1 = Python 카메라 확인 결과 대기 중
*/
volatile uint8_t belt_running = 0;
volatile uint8_t auto_mode = 0;
volatile uint8_t waiting_camera = 0;

/* 현재 흰색 선을 벗어난 뒤 다음 흰색 선 정지를 허용하는 플래그 */
volatile uint8_t wall_released = 0;
uint32_t belt_start_time = 0U;
uint32_t wall_detect_start_time = 0U;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_TIM1_Init(void);

/* USER CODE BEGIN PFP */
void Send_Message(const char *msg);
void Belt_Start(void);
void Belt_Stop(void);
void Send_Check_To_Python(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

void Send_Message(const char *msg)
{
  if (msg != NULL)
  {
    HAL_UART_Transmit(&huart2, (uint8_t*)msg, strlen(msg), 100);
  }
}

void Belt_Start(void)
{
  if (belt_running == 0)
  {
	  belt_running = 1;
	  waiting_camera = 0;
	  wall_released = 0;
	  belt_start_time = HAL_GetTick();
	  wall_detect_start_time = 0U;

    /*
      PA9는 기존 코드처럼 방향 또는 Enable 핀으로 사용.
      모터 방향이 반대면 GPIO_PIN_SET을 GPIO_PIN_RESET으로 바꾸면 됨.
    */
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_9, GPIO_PIN_SET);

    /*
      TIM1 CH1 PWM 시작
      PA8에서 STEP 펄스 자동 출력
    */
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
  }
}

void Belt_Stop(void)
{
  if (belt_running == 1)
  {
    belt_running = 0;

    /*
      TIM1 CH1 PWM 정지
      STEP 펄스 출력 중단
    */
    HAL_TIM_PWM_Stop(&htim1, TIM_CHANNEL_1);

    /*
      STEP 핀 LOW로 정리
    */
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, GPIO_PIN_RESET);
  }
}

void Send_Check_To_Python(void)
{
  waiting_camera = 1;
  Send_Message("CHECK\r\n");
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* MCU Configuration--------------------------------------------------------*/

  HAL_Init();

  SystemClock_Config();

  MX_GPIO_Init();
  MX_USART2_UART_Init();
  MX_TIM1_Init();

  /* USER CODE BEGIN 2 */

  /*
    UART 인터럽트 수신 시작
    Python 또는 시리얼 터미널에서 S, R, X, E, O 입력 수신
  */
  HAL_UART_Receive_IT(&huart2, &rx_data, 1);

  Send_Message(
      "System Ready. Conveyor + Camera Link Mode.\r\n"
      "S: Auto Start\r\n"
      "R: Manual Restart\r\n"
      "X: Stop\r\n"
      "E: Camera Empty / Next Move\r\n"
      "O: Camera Object / Wait\r\n"
      "Stop condition: PA15 wall sensor LOW.\r\n"
  );

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /*
      자동 모드에서 벨트가 움직이는 동안 PA15를 계속 확인함.
      PA15가 흰색 선을 Active-LOW로 감지하면 즉시 벨트를 정지하고
      Python RealSense 쪽에 작업영역 확인을 요청함.
    */
    if (auto_mode == 1 && belt_running == 1)
    {
      uint32_t elapsed_time = HAL_GetTick() - belt_start_time;

      /*
        재출발 후 1초 동안은 PA15 입력을 무시함.
        이후 정지했던 흰색 선 위에서 재시작하자마자 다시 멈추지 않도록,
        현재 선을 벗어나 HIGH가 된 뒤 다음 LOW를 감지했을 때 정지함.
        시간에 의한 강제 정지 조건은 없음.
      */
      if (elapsed_time >= SENSOR_RESTART_IGNORE_MS)
      {
        GPIO_PinState wall_sensor_state = HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_15);

        if (wall_sensor_state == GPIO_PIN_SET)
        {
          wall_released = 1;
          wall_detect_start_time = 0U;
        }
        else if (wall_released == 1)
        {
          if (wall_detect_start_time == 0U)
          {
            wall_detect_start_time = HAL_GetTick();
          }
          else if ((HAL_GetTick() - wall_detect_start_time) >= WALL_DETECT_DEBOUNCE_MS)
          {
            Belt_Stop();
            Send_Message("[STM32] Belt stopped by PA15 wall sensor.\r\n");

          /*
            벨트가 멈췄으니 Python RealSense 쪽에
            작업영역 확인 요청
          */
            Send_Check_To_Python();
          }
        }
      }
    }

    HAL_Delay(5);
  }
  /* USER CODE END WHILE */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI_DIV2;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL16;

  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK |
                                RCC_CLOCKTYPE_SYSCLK |
                                RCC_CLOCKTYPE_PCLK1 |
                                RCC_CLOCKTYPE_PCLK2;

  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{
  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};
  TIM_BreakDeadTimeConfigTypeDef sBreakDeadTimeConfig = {0};

  htim1.Instance = TIM1;

  /*
    기존 설정(Prescaler=113, Period=999)의 STEP 주파수는 약 561.4 Hz.
    Prescaler=94, Period=399로 설정해 약 1684.2 Hz,
    즉 기존의 정확히 3배로 설정.
    1600 pulse/rev 기준으로 약 63.2 rpm.
  */
  htim1.Init.Prescaler = 94;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 399;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;

  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }

  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;

  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }

  if (HAL_TIM_PWM_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }

  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;

  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /*
    Period = 399
    Pulse = 200
    약 50% Duty
  */
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 200;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCNPolarity = TIM_OCNPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  sConfigOC.OCIdleState = TIM_OCIDLESTATE_RESET;
  sConfigOC.OCNIdleState = TIM_OCNIDLESTATE_RESET;

  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }

  /*
    TIM1은 Advanced Timer라 Break/DeadTime 설정 필요
  */
  sBreakDeadTimeConfig.OffStateRunMode = TIM_OSSR_DISABLE;
  sBreakDeadTimeConfig.OffStateIDLEMode = TIM_OSSI_DISABLE;
  sBreakDeadTimeConfig.LockLevel = TIM_LOCKLEVEL_OFF;
  sBreakDeadTimeConfig.DeadTime = 0;
  sBreakDeadTimeConfig.BreakState = TIM_BREAK_DISABLE;
  sBreakDeadTimeConfig.BreakPolarity = TIM_BREAKPOLARITY_HIGH;
  sBreakDeadTimeConfig.AutomaticOutput = TIM_AUTOMATICOUTPUT_DISABLE;

  if (HAL_TIMEx_ConfigBreakDeadTime(&htim1, &sBreakDeadTimeConfig) != HAL_OK)
  {
    Error_Handler();
  }

  HAL_TIM_MspPostInit(&htim1);
}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;

  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_AFIO_CLK_ENABLE();

  /*
    PA15는 STM32F1에서 JTAG 관련 핀과 겹칠 수 있음.
    PA15를 일반 GPIO로 쓰기 위해 JTAG 비활성화, SWD는 유지.
  */
  __HAL_AFIO_REMAP_SWJ_NOJTAG();

  /*
    PA9 초기 LOW
  */
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_9, GPIO_PIN_RESET);

  /*
    PA9: 방향 또는 Enable 출력 핀
  */
  GPIO_InitStruct.Pin = GPIO_PIN_9;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*
    PA15: Active-LOW 센서 입력

    센서 없을 때는 분리해두어도 됨.
    Pull-up 설정으로 기본 HIGH 유지.
    나중에 센서 감지 시 LOW가 들어오면 벨트 정지.
  */
  GPIO_InitStruct.Pin = GPIO_PIN_15;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}

/* USER CODE BEGIN 4 */

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART2)
  {
    /*
      S:
      자동 반복 시작
      Python이 시작할 때 S를 보내면 됨.
    */
    if (rx_data == 'S' || rx_data == 's')
    {
      auto_mode = 1;
      waiting_camera = 0;
      Belt_Start();

      Send_Message("[STM32] Auto mode started. Belt moving.\r\n");
    }

    /*
      R:
      수동 재시작
      자동 모드 상태는 유지한 채 벨트만 다시 시작
    */
    else if (rx_data == 'R' || rx_data == 'r')
    {
      Belt_Start();

      Send_Message("[STM32] Manual restart. Belt moving.\r\n");
    }

    /*
      X:
      전체 정지
      자동 반복 모드도 꺼짐
    */
    else if (rx_data == 'X' || rx_data == 'x')
    {
      auto_mode = 0;
      waiting_camera = 0;
      Belt_Stop();

      Send_Message("[STM32] Stopped by X command.\r\n");
    }

    /*
      E:
      Python 카메라 확인 완료
      지금은 로봇팔이 없으므로 Python이 CHECK를 받으면 바로 E를 보냄.
      E를 받으면 다음 33cm 구간 이동.
    */
    else if (rx_data == 'E' || rx_data == 'e')
    {
      if (auto_mode == 1 && waiting_camera == 1)
      {
        waiting_camera = 0;
        Belt_Start();

        Send_Message("[STM32] E received. Belt restarted.\r\n");
      }
      else
      {
        Send_Message("[STM32] E ignored. Not waiting camera.\r\n");
      }
    }

    /*
      O:
      Python이 객체가 있다고 보낸 경우
      지금은 로봇팔이 없으므로 정지 상태 유지.
      나중에 로봇팔 붙이면 여기서 로봇팔 작업 상태로 넘기면 됨.
    */
    else if (rx_data == 'O' || rx_data == 'o')
    {
      if (auto_mode == 1)
      {
        waiting_camera = 1;
        Belt_Stop();

        Send_Message("[STM32] O received. Object exists. Waiting.\r\n");
      }
    }

    /*
      다음 1바이트 수신 재시작
    */
    HAL_UART_Receive_IT(&huart2, &rx_data, 1);
  }
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  __disable_irq();

  while (1)
  {
  }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
}
#endif
