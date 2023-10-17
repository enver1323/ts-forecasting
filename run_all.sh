scripts=(
    "electricity.sh"
    # "etth1.sh"
    # "etth2.sh"
    "ettm1.sh"
    "ettm2.sh"
    # "exchange.sh"
    # "illness.sh"
    "traffic.sh"
    # "weather.sh"
)

for script in ${scripts[@]};
do
    bash "scripts/${script}"
done