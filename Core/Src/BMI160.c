#include "BMI160.h"

uint8_t BMI160_Initialize( BMI160 *dev, I2C_HandleTypeDef *i2cHandle ) {
	// Set struct parameters
	dev->i2cHandle = i2cHandle;

	dev->acc_mps2[0] = 0.0f;
	dev->acc_mps2[1] = 0.0f;
	dev->acc_mps2[2] = 0.0f;

	dev->gyro_deg[0] = 0.0f;
	dev->gyro_deg[1] = 0.0f;
	dev->gyro_deg[2] = 0.0f;

	// Store number of transaction errors (to be returned at end of function)
	uint8_t errNum = 0;
	HAL_StatusTypeDef status;

	/*
	 * Check device ID (datasheet pg. 49)
	 */
	uint8_t regData;

	status = BMI160_ReadRegister( dev, BMI160_REG_CHIP_ID, &regData );
	errNum += ( status != HAL_OK );

	if ( regData != BMI160_CHIP_ID ) {
		return 255;
	}
}

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
