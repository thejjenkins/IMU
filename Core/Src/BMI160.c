/*
 * BMI160 Accelerometer I2C Driver
 *
 * Author: James Jenkins
 * Date created: Sept 18, 2025
 *
 * This file was written using Phil's Lab as a reference: https://www.youtube.com/watch?v=_JQAve05o_0
 * Datasheet: https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi160-ds000.pdf
 */

#include "BMI160.h"


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
HAL_StatusTypeDef BMI160_ReadRegister( BMI160 *dev, uint8_t reg, uint8_t *data ) {
	return HAL_I2C_Mem_Read( dev->i2cHandle, BMI160_ADDRESS, reg, I2C_MEMADD_SIZE_8BIT, data, 1, HAL_MAX_DELAY );
}
HAL_StatusTypeDef BMI160_ReadRegisters( BMI160 *dev, uint8_t reg, uint8_t *data, uint8_t length ) {
	return HAL_I2C_Mem_Read( dev->i2cHandle, BMI160_ADDRESS, reg, I2C_MEMADD_SIZE_8BIT, data, length, HAL_MAX_DELAY);
}

HAL_StatusTypeDef BMI160_WriteRegister( BMI160 *dev, uint8_t reg, uint8_t *data ) {
	return HAL_I2C_Mem_Write(dev->i2cHandle, BMI160_ADDRESS, reg, I2C_MEMADD_SIZE_8BIT, data, 1, HAL_MAX_DELAY );
}
