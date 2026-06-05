cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(NITorchRegister)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/brudfors/SlicerNitorch")
set(EXTENSION_CATEGORY "Registration")
set(EXTENSION_CONTRIBUTORS "Mikael Brudfors")
set(EXTENSION_DESCRIPTION "GPU-accelerated affine + nonlinear (SVF) 3D image registration powered by the NITorch library, with a Dice validation tab.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/brudfors/SlicerNitorch/main/Resources/Icons/NITorchRegister.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/brudfors/SlicerNitorch/main/SlicerNitorch.png")
set(EXTENSION_DEPENDS "NA")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Scripted module (flat layout: module files live in the repository root)
set(MODULE_NAME NITorchRegister)

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  NITorchRegisterLib/__init__.py
  NITorchRegisterLib/registration.py
  NITorchRegisterLib/grid_transform.py
  NITorchRegisterLib/validation.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  )

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
