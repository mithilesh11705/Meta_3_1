# Pre-Submission Validation Checklist

## Automated Validation Tests

### 1. HF Space Deployment Test
```bash
# Test that the app responds to health checks
curl -f http://localhost:7860/health

# Test reset endpoint works
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task":"easy"}'

# Expected: 200 response and proper Observation object
```

### 2. OpenEnv Spec Compliance
```bash
# Validate openenv.yaml
python -c "import yaml; yaml.safe_load(open('openenv.yaml'))"

# Validate typed models
python -c "from pr_review_env.models import Action, Observation; print('Models OK')"

# Test required endpoints
python -c "
from pr_review_env.env import PRReviewEnv
env = PRReviewEnv()
obs = env.reset('easy')      # reset() works
action = Action(decision='approve', labels=['bug'], priority='low', review_summary='test')
result = env.step(action)    # step() works
state = env.get_state()      # state() works
print('All endpoints work')
"
```

### 3. Dockerfile Build Test
```bash
# Build should complete successfully
docker build -t pr-review-env .

# Should run without errors
docker run --rm -p 7860:7860 pr-review-env
```

### 4. Baseline Reproduction Test
```bash
# Set required environment variables
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export OPENAI_API_KEY=your_token_here

# Run inference - must complete without error
python inference.py

# Must output exact format:
# [START] task=easy env=pr-review-env model=...
# [STEP] step=1 action=... reward=0.00 done= error=null
# [END] success=true steps=1 score=0.00 rewards=...
```

### 5. Tasks + Graders Test
```bash
# Test all 3 tasks have working graders
python -c "
from pr_review_env.tasks.easy import grade as easy_grade
from pr_review_env.tasks.medium import grade as medium_grade  
from pr_review_env.tasks.hard import grade as hard_grade
from pr_review_env.models import Action

# Test each grader returns strict (0,1) range
actions = [
    Action(decision='approve', labels=['bug'], priority='low', review_summary='LGTM'),
    Action(decision='request_changes', labels=['security'], priority='critical', review_summary='Security issue'),
    Action(decision='request_changes', labels=['bug'], priority='high', review_summary='Race condition')
]

scores = [easy_grade(actions[0]), medium_grade(actions[1]), hard_grade(actions[2])]
print(f'Scores: {scores}')
print(f'All in range strict (0,1): {all(0.0 < s < 1.0 for s in scores)}')
"
```

## Environment Variables Check

### Required Variables
```bash
# Must be set before running inference
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct  
export OPENAI_API_KEY=your_actual_token

# Verify they're set
env | grep -E "(API_BASE_URL|MODEL_NAME|OPENAI_API_KEY|HF_TOKEN)"
```

## Performance Requirements

### Runtime < 20 minutes
```bash
# Time the inference script
time python inference.py

# Should complete in well under 20 minutes
```

### Resource Limits (vCPU=2, memory=8GB)
```bash
# Monitor resource usage during inference
# Should stay under 2 CPU cores and 8GB RAM
```

## Output Format Validation

### Required stdout format
```bash
# Must exactly match this pattern:
[START] task=easy env=pr-review-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"decision":"approve","labels":["bug"],"priority":"low","review_summary":"LGTM"} reward=0.95 done=true error=null
[END] success=true steps=1 score=0.95 rewards=0.95

# No deviation in field names, ordering, or formatting
```

## File Structure Check

### Required files in root directory
```bash
# Must exist in project root:
ls -la | grep -E "(inference.py|openenv.yaml|Dockerfile|requirements.txt)"

# inference.py must be in root (not in subdirectory)
test -f ./inference.py && echo "inference.py in root" || echo "ERROR: inference.py not in root"
```

## OpenAI Client Usage

### Must use OpenAI client (not requests)
```bash
# Check inference.py uses OpenAI client
grep -n "from openai import OpenAI" inference.py
grep -n "OpenAI(" inference.py

# Should not use requests module for LLM calls
grep -v "import requests" inference.py | head -5
```

## Final Validation Command

### Run all checks at once
```bash
#!/bin/bash
echo "=== Pre-Submission Validation ==="

# 1. Check required files exist
echo "Checking required files..."
required_files=("inference.py" "openenv.yaml" "Dockerfile" "requirements.txt")
for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "  $file exists"
    else
        echo "  ERROR: $file missing"
        exit 1
    fi
done

# 2. Check environment variables
echo "Checking environment variables..."
if [[ -z "$API_BASE_URL" ]] || [[ -z "$MODEL_NAME" ]] || { [[ -z "$OPENAI_API_KEY" ]] && [[ -z "$HF_TOKEN" ]]; }; then
    echo "  ERROR: Missing required environment variables"
    echo "  Set: API_BASE_URL, MODEL_NAME, OPENAI_API_KEY (or HF_TOKEN)"
    exit 1
else
    echo "  Environment variables set"
fi

# 3. Test Python imports
echo "Testing Python imports..."
python -c "
import yaml
from pr_review_env.models import Action, Observation
from pr_review_env.env import PRReviewEnv
from inference import main
print('  All imports successful')
" || exit 1

# 4. Test graders
echo "Testing graders..."
python -c "
from pr_review_env.tasks.easy import grade as easy_grade
from pr_review_env.tasks.medium import grade as medium_grade
from pr_review_env.tasks.hard import grade as hard_grade
from pr_review_env.models import Action

scores = [
    easy_grade(Action(decision='approve', labels=['bug'], priority='low', review_summary='LGTM')),
    medium_grade(Action(decision='request_changes', labels=['security'], priority='critical', review_summary='Security issue')),
    hard_grade(Action(decision='request_changes', labels=['bug'], priority='high', review_summary='Race condition'))
]
print(f'  Scores: {scores}')
print(f'  All in strict (0,1) range: {all(0.0 < s < 1.0 for s in scores)}')
" || exit 1

# 5. Test server endpoints
echo "Testing server endpoints..."
python -c "
from pr_review_env.env import PRReviewEnv
env = PRReviewEnv()
obs = env.reset('easy')
result = env.step(Action(decision='approve', labels=['bug'], priority='low', review_summary='test'))
state = env.get_state()
print('  All endpoints functional')
" || exit 1

echo "=== All validations passed! ==="
echo "Ready for submission."
```

## Common Issues to Fix

### 1. Docker build fails
- Check Dockerfile syntax
- Ensure requirements.txt is valid
- Verify all files are copied correctly

### 2. Inference script fails
- Check environment variables are set
- Verify OpenAI client usage
- Ensure output format is exact

### 3. Graders return invalid scores
- Check reward computation logic
- Verify score bounds (strict (0,1))
- Test with various actions

### 4. Server doesn't respond
- Check FastAPI app structure
- Verify endpoint implementations
- Test with curl commands

## Ready for Submission

When all validations pass:
1. Ensure HF Space is deployed and responds to health checks
2. Verify `/reset`, `/step`, `/tasks`, and `/health` all respond on the live Space
3. Verify openenv.yaml is valid
4. Confirm Dockerfile builds successfully
5. Test inference script completes without errors
6. Verify all 3 tasks have working graders with proper score ranges
7. Check environment variables are properly configured

Run the final validation command above to confirm everything is ready.
