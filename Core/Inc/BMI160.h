/*
 * BMI160 Accelerometer I2C Driver
 *
 * Author: James Jenkins
 * Date created: Sept 18, 2025
 *
 * This file was written using Phil's Lab as a reference: https://www.youtube.com/watch?v=_JQAve05o_0
 * Datasheet: https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi160-ds000.pdf
 */

#ifndef BMI160_I2C_DRIVER_H
#define BMI160_I2C_DRIVER_H

#include "stm32l4xx.h" // Needed for I2C

/*
 * DEFINES
 */
#define BMI160_ADDRESS				(0x69 << 1) // SDO = 0 -> 0x68, SDO = 1 -> 0x69 (p.93)
#define BMI160_CHIP_ID				0xD1

/*
 * REGISTERS - READ ONLY
 */
#define BMI160_REG_
#define BMI160_REG_CHIP_ID			0x00
#define BMI160_REG_ERR_REG			0x02
#define BMI160_REG_PMU_STATUS		0x03
#define BMI160_REG_MAG_X_7_0		0x04
#define BMI160_REG_MAG_X_15_8		0x05
#define BMI160_REG_MAG_Y_7_0		0x06
#define BMI160_REG_MAG_Y_15_8		0x07
#define BMI160_REG_MAG_Z_7_0		0x08
#define BMI160_REG_MAG_Z_15_8		0x09
#define BMI160_REG_RHALL_7_0		0x0A
#define BMI160_REG_RHALL_15_8		0x0B
#define BMI160_REG_GYR_X_7_0		0x0C
#define BMI160_REG_GYR_X_15_8		0x0D
#define BMI160_REG_GYR_Y_7_0		0x0E
#define BMI160_REG_GYR_Y_15_8		0x0F
#define BMI160_REG_GYR_Z_7_0		0x10
#define BMI160_REG_GYR_Z_15_8		0x11
#define BMI160_REG_ACC_X_7_0		0x12
#define BMI160_REG_ACC_X_15_8		0x13
#define BMI160_REG_ACC_Y_7_0		0x14
#define BMI160_REG_ACC_Y_15_8		0x15
#define BMI160_REG_ACC_Z_7_0		0x16
#define BMI160_REG_ACC_Z_15_8		0x17
#define BMI160_REG_SENSORTIME_0		0x18
#define BMI160_REG_SENSORTIME_1		0x19
#define BMI160_REG_SENSORTIME_2		0x1A
#define BMI160_REG_STATUS			0x1B
#define BMI160_REG_INT_STATUS_0		0x1C
#define BMI160_REG_INT_STATUS_1		0x1D
#define BMI160_REG_INT_STATUS_2		0x1E
#define BMI160_REG_INT_STATUS_3		0x1F
#define BMI160_REG_TEMPERATURE_0	0x20
#define BMI160_REG_TEMPERATURE_1	0x21
#define BMI160_REG_FIFO_LENGTH_0	0x22
#define BMI160_REG_FIFO_LENGTH_1	0x23
#define BMI160_REG_FIFO_DATA		0x24
#define BMI160_REG_STEP_CNT_7_0		0x78
#define BMI160_REG_STEP_CNT_15_8	0x79

/*
 * REGISTERS - READ/WRITE
 */
#define BMI160_REG_ACC_CONF			0x40
#define BMI160_REG_ACC_RANGE		0x41
#define BMI160_REG_GYR_CONF			0x42
#define BMI160_REG_GYR_RANGE		0x43
#define BMI160_REG_MAG_CONF			0x44
#define BMI160_REG_FIFO_DOWNS		0x45
#define BMI160_REG_FIFO_CONFIG_0	0x46
#define BMI160_REG_FIFO_CONFIG_1	0x47
#define BMI160_REG_MAG_IF_0			0x4B
#define BMI160_REG_MAG_IF_1			0x4C
#define BMI160_REG_MAG_IF_2			0x4D
#define BMI160_REG_MAG_IF_3			0x4E
#define BMI160_REG_MAG_IF_4			0x4F
#define BMI160_REG_INT_EN_0			0x50
#define BMI160_REG_INT_EN_1			0x51
#define BMI160_REG_INT_EN_2			0x52
#define BMI160_REG_INT_OUT_CTRL		0x53
#define BMI160_REG_INT_LATCH		0x54
#define BMI160_REG_INT_MAP_0		0x55
#define BMI160_REG_INT_MAP_1		0x56
#define BMI160_REG_INT_MAP_2		0x57
#define BMI160_REG_INT_DATA_0		0x58
#define BMI160_REG_INT_DATA_1		0x59
#define BMI160_REG_INT_LOWHIGH_0	0x5A
#define BMI160_REG_INT_LOWHIGH_1	0x5B
#define BMI160_REG_INT_LOWHIGH_2	0x5C
#define BMI160_REG_INT_LOWHIGH_3	0x5D
#define BMI160_REG_INT_LOWHIGH_4	0x5E
#define BMI160_REG_INT_MOTION_0		0x5F
#define BMI160_REG_INT_MOTION_1		0x60
#define BMI160_REG_INT_MOTION_2		0x61
#define BMI160_REG_INT_MOTION_3		0x62
#define BMI160_REG_INT_TAP_0		0x63
#define BMI160_REG_INT_TAP_1		0x64
#define BMI160_REG_INT_ORIENT_0		0x65
#define BMI160_REG_INT_ORIENT_1		0x66
#define BMI160_REG_INT_FLAT_0		0x67
#define BMI160_REG_INT_FLAT_1		0x68
#define BMI160_REG_FOC_CONF			0x69
#define BMI160_REG_CONF				0x6A
#define BMI160_REG_IF_CONF			0x6B
#define BMI160_REG_PMU_TRIGGER		0x6C
#define BMI160_REG_SELF_TEST		0x6D
#define BMI160_REG_NV_CONF			0x70
#define BMI160_REG_OFFSET_0			0x71
#define BMI160_REG_OFFSET_1			0x72
#define BMI160_REG_OFFSET_2			0x73
#define BMI160_REG_OFFSET_3			0x74
#define BMI160_REG_OFFSET_4			0x75
#define BMI160_REG_OFFSET_5			0x76
#define BMI160_REG_OFFSET_6			0x77
#define BMI160_REG_STEP_CONF_0		0x7A
#define BMI160_REG_STEP_CONF_1		0x7B
#define BMI160_REG_CMD				0x7E

/*
 * SENSOR STRUCT
 */
typedef struct {
	// I2C handle
	I2C_HandleTypeDef *i2cHandle;

	// Acceleration data (x,y,z) in m/s^2
	float acc_mps2[3];
}BMI160;

/*
 * INITIALIZATION
 */
uint8_t BMI160_Initialize( BMI160 *dev, I2C_HandleTypeDef *i2cHandle );

/*
 * DATA ACQUISITION
 */
HAL_StatusTypeDef BMI160_ReadAccelerations( BMI160 *dev );
HAL_StatusTypeDef BMI160_ReadGyro( BMI160 *dev );

/*
 * LOW-LEVEL FUNCTIONS
 */
HAL_StatusTypeDef BMI160_ReadRegister( BMI160 *dev, uint8_t reg, uint8_t *data );
HAL_StatusTypeDef BMI160_ReadRegisters( BMI160 *dev, uint8_t reg, uint8_t *data, uint8_t length );

HAL_StatusTypeDef BMI160_WriteRegister( BMI160 *dev, uint8_t reg, uint8_t *data );

#endif
