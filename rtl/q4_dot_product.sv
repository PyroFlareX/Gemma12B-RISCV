// One-cycle, ready/valid signed Q4 x signed INT8 dot-product unit.
// lane 0 occupies q4_values[3:0] and activations[7:0]. The 32 products are
// accumulated exactly into a signed 32-bit result. This is a custom accelerator
// datapath and does not implement the RISC-V Vector (RVV) ISA.
module q4_dot_product #(
    parameter int LANES = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    in_valid,
    output logic                    in_ready,
    input  logic [LANES*4-1:0]      q4_values,
    input  logic [LANES*8-1:0]      activations,
    output logic                    out_valid,
    input  logic                    out_ready,
    output logic signed [31:0]      out_sum
);
    integer lane;
    logic signed [3:0] q4_lane;
    logic signed [7:0] activation_lane;
    logic signed [11:0] product;
    logic signed [31:0] combinational_sum;

    always_comb begin
        combinational_sum = 32'sd0;
        q4_lane = 4'sd0;
        activation_lane = 8'sd0;
        product = 12'sd0;
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            q4_lane = $signed(q4_values[lane*4 +: 4]);
            activation_lane = $signed(activations[lane*8 +: 8]);
            product = $signed({{8{q4_lane[3]}}, q4_lane}) *
                      $signed({{4{activation_lane[7]}}, activation_lane});
            combinational_sum = combinational_sum + {{20{product[11]}}, product};
        end
    end

    assign in_ready = !out_valid || out_ready;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_sum <= 32'sd0;
        end else if (in_ready) begin
            out_valid <= in_valid;
            if (in_valid)
                out_sum <= combinational_sum;
        end
    end
endmodule
