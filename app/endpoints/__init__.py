from fastapi import APIRouter
from endpoints.hazzard_detection import router as hazzard_detection_router
from endpoints.sign_detection import router as sign_detection_router
from endpoints.smolvlm import router as smolvlm_router
from endpoints.paddleocr import router as paddleocr_router
from endpoints.model_management import router as model_management_router

router = APIRouter()

# Thêm các router con vào router chính
router.include_router(hazzard_detection_router, prefix="/hazzard_detect", tags=["hazzard_detect"])
router.include_router(sign_detection_router, prefix="/sign_detect", tags=["sign_detect"])
router.include_router(smolvlm_router, prefix="/smolvlm", tags=["smolvlm"])
router.include_router(paddleocr_router, prefix="/paddleocr", tags=["paddleocr"])
router.include_router(model_management_router, prefix="/models", tags=["model_management"])

# router.include_router(advise.router, prefix = "/advise", tags=["advise"])
# router.include_router(quiz.router, prefix = "/quiz", tags=["quiz"])
# router.include_router(uni_info.router, prefix = "/uni_info", tags=["uni_info"])
# router.include_router(auth.router, prefix="/auth", tags=["auth"])
# router.include_router(advise.router, prefix="/advise", tags=["advise"])