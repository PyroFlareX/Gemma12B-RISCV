// Streaming one row of block-Q4 weights against a resident INT8 activation row.
// Each group occupies five little-endian 32-bit words: four Q4 payload words
// (eight low-nibble-first signed codes each), then a signed Q16.16 scale word.
// The result is signed Q16.16 in a 64-bit accumulator; no rounding is applied.
module q4_gemv_tile #(
    parameter integer GROUPS_PER_ROW = 1
) (
    input  logic                                      clk,
    input  logic                                      rst_n,
    input  logic                                      start,
    input  logic [31:0]                               row,
    input  logic [GROUPS_PER_ROW*32*8-1:0]            activations,
    output logic                                      done,
    output logic signed [63:0]                        result,
    output logic                                      mem_req,
    output logic [31:0]                               mem_addr,
    input  logic                                      mem_ready,
    input  logic [31:0]                               mem_rdata
);
    localparam integer ROW_BYTES = GROUPS_PER_ROW * 20;
    localparam integer GROUP_INDEX_WIDTH = (GROUPS_PER_ROW <= 1) ? 1 : $clog2(GROUPS_PER_ROW);

    typedef enum logic [1:0] {ST_IDLE, ST_READ_Q4, ST_READ_SCALE, ST_DOT} state_t;
    state_t state;

    logic [31:0] row_base_addr;
    logic [GROUPS_PER_ROW*32*8-1:0] activation_reg;
    logic [GROUP_INDEX_WIDTH-1:0] group_index;
    logic [2:0] word_index;
    logic [31:0] q4_words [0:3];
    logic signed [31:0] scale_reg;
    logic signed [63:0] accumulator;

    integer lane;
    logic signed [3:0] q4_lane;
    logic signed [7:0] activation_lane;
    logic signed [11:0] product_lane;
    logic signed [31:0] group_dot;
    logic signed [63:0] scaled_group;

    always @* begin
        mem_req = (state == ST_READ_Q4) || (state == ST_READ_SCALE);
        if (state == ST_READ_Q4)
            mem_addr = row_base_addr + ((group_index * 5 + word_index) * 4);
        else if (state == ST_READ_SCALE)
            mem_addr = row_base_addr + ((group_index * 5 + 4) * 4);
        else
            mem_addr = 32'd0;
    end

    always @* begin
        group_dot = 32'sd0;
        q4_lane = 4'sd0;
        activation_lane = 8'sd0;
        product_lane = 12'sd0;
        for (lane = 0; lane < 32; lane = lane + 1) begin
            q4_lane = $signed(q4_words[lane / 8][(lane % 8) * 4 +: 4]);
            activation_lane = $signed(activation_reg[(group_index * 32 + lane) * 8 +: 8]);
            product_lane = $signed({{8{q4_lane[3]}}, q4_lane}) *
                           $signed({{4{activation_lane[7]}}, activation_lane});
            group_dot = group_dot + {{20{product_lane[11]}}, product_lane};
        end
        scaled_group = $signed({{32{group_dot[31]}}, group_dot}) *
                       $signed({{32{scale_reg[31]}}, scale_reg});
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            row_base_addr <= 32'd0;
            activation_reg <= '0;
            group_index <= '0;
            word_index <= 3'd0;
            scale_reg <= 32'sd0;
            accumulator <= 64'sd0;
            result <= 64'sd0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                ST_IDLE: begin
                    if (start) begin
                        row_base_addr <= row * ROW_BYTES;
                        activation_reg <= activations;
                        group_index <= '0;
                        word_index <= 3'd0;
                        accumulator <= 64'sd0;
                        state <= ST_READ_Q4;
                    end
                end
                ST_READ_Q4: begin
                    if (mem_ready) begin
                        q4_words[word_index] <= mem_rdata;
                        if (word_index == 3'd3) begin
                            word_index <= 3'd0;
                            state <= ST_READ_SCALE;
                        end else begin
                            word_index <= word_index + 1'b1;
                        end
                    end
                end
                ST_READ_SCALE: begin
                    if (mem_ready) begin
                        scale_reg <= $signed(mem_rdata);
                        state <= ST_DOT;
                    end
                end
                ST_DOT: begin
                    accumulator <= accumulator + scaled_group;
                    if (group_index == GROUPS_PER_ROW - 1) begin
                        result <= accumulator + scaled_group;
                        done <= 1'b1;
                        state <= ST_IDLE;
                    end else begin
                        group_index <= group_index + 1'b1;
                        word_index <= 3'd0;
                        state <= ST_READ_Q4;
                    end
                end
                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule
